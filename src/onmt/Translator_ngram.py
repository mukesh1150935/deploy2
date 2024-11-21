import onmt
# import lm
import torch.nn as nn
import torch
from torch.autograd import Variable
from copy import deepcopy
import time
import pickle
import os
import re

def isNumeral(s):
    x = re.findall(r'[0-9]+[\.,][0-9]+',s)
    if len(x) >= 1:
        return True
    return False

class Translator_ngram(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model,map_location=lambda storage, loc: storage)

        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']

        encoder = onmt.Models.Encoder(model_opt, self.src_dict, model_opt.fix_src_emb, use_fert=True)
        decoder = onmt.Models.Decoder(model_opt, self.tgt_dict, model_opt.tie_emb)
        model = onmt.Models.NMTModel(encoder, decoder)

        if not model_opt.nonlin_gen:
            generator = nn.Sequential(nn.Linear(model_opt.rnn_size, model_opt.output_emb_size))
        else:
            generator = nn.Sequential(nn.Linear(model_opt.rnn_size, model_opt.output_emb_size), nn.ReLU(), nn.Linear(model_opt.output_emb_size, model_opt.output_emb_size))

        if opt.tgt_dict is not None:
            self.tgt_dict = torch.load(opt.tgt_dict)['dicts']['tgt']

        # print(self.tgt_dict.embeddings[0][:10])
        # print(self.tgt_dict.unigram_embeddings[0][:10])
        # print(self.tgt_dict.embeddings[self.tgt_dict.size_uni()][:10])
        # print(self.tgt_dict.ngram_embeddings[0][:10])


        target_embeddings = nn.Embedding(self.tgt_dict.size(), model_opt.output_emb_size)
        norm = self.tgt_dict.embeddings.norm(p=2, dim=1, keepdim=True)+1e-6
        target_embeddings.weight.data.copy_(self.tgt_dict.embeddings.div(norm))
        target_embeddings.weight.requires_grad=False

        target_uni_embeddings = nn.Embedding(self.tgt_dict.size_uni(), model_opt.output_emb_size)
        target_uni_embeddings.weight.data.copy_(self.tgt_dict.unigram_embeddings.div(norm[:self.tgt_dict.size_uni()]))
        target_uni_embeddings.weight.requires_grad=False

        target_ngram_embeddings = nn.Embedding(self.tgt_dict.size_ngram(), model_opt.output_emb_size)
        target_ngram_embeddings.weight.data.copy_(self.tgt_dict.ngram_embeddings.div(norm[self.tgt_dict.size_uni():]))
        target_ngram_embeddings.weight.requires_grad=False

        encoder_state_dict = [('encoder.'+k,v) for k, v in checkpoint['encoder'].items()]
        decoder_state_dict = [('decoder.'+k,v) for k, v in checkpoint['decoder'].items() if not k.startswith("word_emb")]
        model_state_dict = dict(encoder_state_dict+decoder_state_dict)

        model.load_state_dict(model_state_dict)
        generator.load_state_dict(checkpoint['generator'])

        if model_opt.tie_emb:
            target_embeddings.weight.data.copy_(model.decoder.word_lut.weight.data)

        if opt.cuda:
            model.cuda()
            generator.cuda()
            target_embeddings.cuda()
            target_uni_embeddings.cuda()
            target_ngram_embeddings.cuda()
        else:
            model.cpu()
            generator.cpu()
            target_embeddings.cpu()
            target_uni_embeddings.cpu()
            target_ngram_embeddings.cpu()

        model.generator = generator

        self.model = model
        self.target_embeddings = target_embeddings
        self.target_uni_embeddings = target_uni_embeddings
        self.target_ngram_embeddings = target_ngram_embeddings
        self.model.eval()

        if self.opt.lookup_dict:
            self.lookup_dict = eval(open(self.opt.lookup_dict).read())
        else:
            self.lookup_dict = None

        if self.opt.use_lm:
            self.LangModel, _, _, _ = torch.load(self.opt.saved_lm)
        
        self.fert_dim = model_opt.fert_dim


    def buildData(self, srcBatch, goldBatch, alignBatch=None, tgtUniBatch=None):
        srcData = [self.src_dict.convertToIdx(b,
                    onmt.Constants.UNK_WORD)[0] for b in srcBatch]
        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                       onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD,
                       onmt.Constants.EOS_WORD)[0] for b in goldBatch]

        return onmt.Dataset(srcData, tgtData, tgtUniBatch, alignBatch, 
            self.opt.batch_size, self.opt.cuda, volatile=True)

    def replace_unk(self, i, tok, attn, src, tokens=None):
        _, maxIndex = attn.max(0)
        if tok == onmt.Constants.UNK_WORD:
            _, maxIndex = attn.topk(2, 0)
            # print (tokens[i],)
            dist = (maxIndex[0]-i)*1.0/len(src)
            if src[maxIndex[0]] in [",", "."] or dist > 0.5 or dist < -0.5:
                mI = maxIndex[1]
                # print ("Halelujah")
                # input("Hello")
            else:
                mI = maxIndex[0]

            while mI>0 and src[mI-1].endswith("@@"):
                mI -= 1

            if self.lookup_dict is not None:
                t = src[mI]
                if src[mI].endswith("@@"):
                    t = src[mI]
                    k = mI+1
                    while k < len(src) and src[k].endswith("@@"):
                        t += " " + src[k]
                        k+=1
                    if k < len(src):
                        t += " " + src[k]

                    t = t.replace("@@", "").replace(" ", "")
                    if t in self.lookup_dict:
                        tok = self.lookup_dict[t]
                    else:
                        tok = t
                elif src[mI] in self.lookup_dict:
                    tok = self.lookup_dict[src[mI]]
                else:
                    tok = src[mI]
            else:
                t = ""
                if src[mI].endswith("@@"):
                    t = src[mI]
                    k = mI+1
                    while k < len(src) and src[k].endswith("@@"):
                        t += " " + src[k]
                        k+=1
                    if k < len(src):
                        t += " " + src[k]
                    tok = t
                else:
                    tok = src[mI]

        if isNumeral(tok):
            mI = maxIndex[0]
            while mI>0 and src[mI-1].endswith("@@"):
                mI -= 1
            t = src[mI]
            if src[mI].endswith("@@"):
                t = src[mI]
                k = mI+1
                while k < len(src) and src[k].endswith("@@"):
                    t += " " + src[k]
                    k+=1
                if k < len(src):
                    t += " " + src[k]
                tokens[i] = t
            if tokens:
                if i+1<len(tokens) and tokens[i+1] in ["a.m.", "p.m."]:
                    tok = t.replace("."," : ")
                else:
                    t = t.replace(".","]")
                    t = t.replace(",",".")
                    tok = t.replace("]", ",")
        if tok == '%':
            if src[maxIndex[0]] == "Prozent":
                tok = "percent"
        return tok

    def buildTargetTokens(self, pred, src, attn):
        # tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS_WORD)
        tokens = tokens[:-1]  # EOS
        # print tokens
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                tokens[i] = self.replace_unk(i, tokens[i], attn[i], src, tokens)
        # else:
            # tokens[i] = ""

        return tokens

    def _get_scores(self, out, target_embeddings=None):

        if self.opt.loss == "l2":
            rA = (out*out).sum(dim=1)
            rA = rA.unsqueeze(dim=1)
            B = target_embeddings.weight
            rB = (B*B).sum(dim=1)
            rB[0].data+=10.0
            rB[2].data+=10.0
            rB = rB.unsqueeze(dim=0)

            M = 2*out.matmul(B.t())
            dists = rA-M
            dists = dists+rB
            return -dists

        elif self.opt.loss == 'nllvmf':
            norm = torch.log(1+out.norm(p=2, dim=-1, keepdim=True))
            # norm = out.norm(p=2, dim=-1, keepdim=True)
            logcmk = onmt.Logcmk.apply
            if target_embeddings:
                return logcmk(norm) + out.matmul(self.target_embeddings.weight.t())
            else:
                return (logcmk(norm) + out.matmul(self.target_uni_embeddings.weight.t())), (logcmk(norm) + out.matmul(self.target_ngram_embeddings.weight.t()))

        else:
            out_n = torch.nn.functional.normalize(out, p=2, dim=-1)
            return out_n.matmul(self.target_embeddings.weight.t())

    def get_LM_logprob(self, logit, target):
        return 1

    def translateBatch(self, srcBatch, tgtBatch, alignBatch):

        batchSize = srcBatch[0].size(1)
        beamSize = self.opt.beam_size
        knntime = 0.0
        #  (1) run the encoder on the src
        encStates, context, fert = self.model.encoder(srcBatch, is_fert=True)
        init_fert = deepcopy(fert)
        init_fert = torch.max(init_fert, dim=-1)[1].float()
        # cov = torch.max(cov, dim=-1)[1].float()
        srcBatch = srcBatch[0] # drop the lengths needed for encoder

        rnnSize = context.size(2)
        encStates = (self.model._fix_enc_hidden(encStates[0]),
                      self.model._fix_enc_hidden(encStates[1]))

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = srcBatch.data.eq(onmt.Constants.PAD).t()
        def applyContextMask(m):
            if isinstance(m, onmt.modules.GlobalAttention):
                m.applyMask(padMask)
            elif isinstance(m, onmt.modules.GlobalAttentionOriginal):
                m.applyMask(padMask)


        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        # decStates = encStates
        if self.opt.use_lm:
            lm_hidden = self.LangModel.initialize_hidden(1, batchSize)

        context = Variable(context.data.repeat(1, beamSize, 1))
        # cov = Variable(torch.zeros((context.size(1),context.size(0))), requires_grad=True)
        # cov = cov.cuda()
        decStates = (Variable(encStates[0].data.repeat(1, beamSize, 1)),
                     Variable(encStates[1].data.repeat(1, beamSize, 1)))

        beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]

        decOut = self.model.make_init_decoder_output(context)

        padMask = srcBatch.data.eq(onmt.Constants.PAD).t().unsqueeze(0).repeat(beamSize, 1, 1)

        batchIdx = list(range(batchSize))
        previous_batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.opt.max_sent_length):

            self.model.decoder.apply(applyContextMask)

            # Prepare decoder input.
            inputs = []
            for i_b, b in enumerate(beam):
                if not b.done():
                    c = b.getCurrentState()
                    if self.opt.replace_unk and i != 0 and (int(c)==onmt.Constants.UNK or int(c)== onmt.Constants.UNK + self.tgt_dict.size_uni()):
                        tok = self.tgt_dict.getLabel(int(c))
                        if tok == onmt.Constants.UNK_WORD:
                            b_i = previous_batchIdx[i_b]
                            _src = self.src_dict.convertToLabels(srcBatch.data.t()[i_b], onmt.Constants.PAD_WORD)
                            tok = self.replace_unk(i, tok, attn[b_i][0].data, _src)
                            if tok in self.tgt_dict.labelToIdx.keys():
                                c = torch.LongTensor([self.tgt_dict.labelToIdx[tok]]).cuda()
                    inputs.append(c)
            input_ = torch.stack(inputs).t().contiguous().view(1, -1)
            # input_ = torch.stack([b.getCurrentState() for b in beam
            #                    if not b.done()]).t().contiguous().view(1, -1)
            input_var = Variable(input_, volatile=True)
            # decOut, decStates, attn, alignBatch = self.model.decoder(
            #     input_var, decStates, context, decOut, alignBatch)
            decOut, decStates, attn = self.model.decoder(
                input_var, decStates, context, decOut)

            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.squeeze(0)
            out = self.model.generator.forward(decOut)

            if self.opt.use_lm:
                lm_output, lm_hidden, _, _ = self.LangModel(input_var, lm_hidden)
                lm_output = torch.log(lm_output+1e-12)

            beg = time.time()
            original_scores = self._get_scores(out, self.target_embeddings)
            uni_scores, ngram_scores = self._get_scores(out)
            # print(fert)
            # print(attn.shape)
            fert_combined = torch.bmm(attn.unsqueeze(1)[:,:,:-1], fert[:,:-1]).squeeze(1)
            # print(fert_combined)

            if self.fert_dim > 2:
                uni_prob = torch.sum(fert_combined[:,:2], dim=1).unsqueeze(1)
                ngram_prob = torch.sum(fert_combined[:,2:], dim=1).unsqueeze(1)
            else:
                uni_prob = fert_combined[:,0].unsqueeze(1)
                ngram_prob = fert_combined[:,1].unsqueeze(1)
            scores = torch.cat((uni_scores*uni_prob, ngram_scores * ngram_prob),dim=1)
            # print(uni_prob.shape)
            # print(ngram_prob.shape)
            # print(uni_scores.shape)
            # print(ngram_scores.shape)
            # print(uni_prob.shape)
            # print(ngram_prob.shape)
            # print(scores.shape)
            # print(self.target_embeddings.weight.shape)
            # print(self.target_uni_embeddings.weight.shape)
            # print(self.target_ngram_embeddings.weight.shape)
            topk = scores.topk(beamSize, 1, True, True)[1].cpu().data.numpy()
            topk_uni = uni_scores.topk(beamSize, 1, True, True)[1].cpu().data.numpy()
            topk_ngram = ngram_scores.topk(beamSize, 1, True, True)[1].cpu().data.numpy()
            topk_original = original_scores.topk(beamSize, 1, True, True)[1].cpu().data.numpy()
            # print(topk)
            # print(topk_original)
            # print(topk_uni)
            # print(topk_ngram)
            # print(scores[0][:10])
            # raise
            # print(uni_prob[:20])
            # print(ngram_prob[:20])
            # print(scores)
            # raise

            diff = time.time()-beg
            knntime += diff

            if self.opt.use_lm:
                scores += 0.2*lm_output.squeeze(0)

            wordLk = scores.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            out = out.view(remainingSents, beamSize, -1).contiguous()
            fert_beam = fert_combined.view(remainingSents, beamSize, -1).contiguous()
            init_fert_beam = init_fert.view(remainingSents, beamSize, -1).contiguous()

            active = []
            for b in range(batchSize):
                if beam[b].done():
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx], out.data[idx], fert_beam.data[idx], init_fert_beam.data[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(
                        -1, beamSize, remainingSents, decState.size(2))[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(1, beam[b].getCurrentOrigin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            previous_batchIdx = batchIdx
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx) \
                                    .view(*newSize), volatile=True)

            decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)
            padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)

        #  (4) package everything up

        allHyp, allScores, allAttn, allOut, allOutScores, allFert, allInitFert = [], [], [], [], [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            valid_attn = srcBatch.data[:, b].ne(onmt.Constants.PAD).nonzero().squeeze(1)

            hyps, attn, out, score, fert, init_fert = zip(*[beam[b].getHyp(times, k) for (times, k) in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]

            allHyp += [hyps]
            allAttn += [attn]
            allOut += [out]
            allOutScores += [score]
            allFert += [fert]
            allInitFert += [init_fert]

        return allHyp, allScores, allAttn, allOut, allOutScores, allFert, allInitFert, knntime

    def translate(self, srcBatch, goldBatch, alignBatch=None):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        src, tgt, tgt_uni, align, indices = dataset[0]

        #  (2) translate
        pred, predScore, attn, out, outScore, cov, init_fert, knntime = self.translateBatch(src, tgt, align)
        pred, predScore, attn, out, outScore, cov, init_fert = list(zip(*sorted(zip(pred, predScore, attn, out, outScore, cov, init_fert, indices), key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n]) for n in range(self.opt.n_best) ]
            )

        return predBatch, predScore, knntime, attn, out, outScore, cov, init_fert
