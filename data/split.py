def split_sentences(input_file, tamil_output_file, english_output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(tamil_output_file, 'w', encoding='utf-8') as tamil_file, \
         open(english_output_file, 'w', encoding='utf-8') as english_file:

        for line in infile:
            # Split line by tab character
            parts = line.strip().split('\t')
            if len(parts) == 2:
                tamil_sentence, english_sentence = parts
                tamil_file.write(tamil_sentence + '\n')
                english_file.write(english_sentence + '\n')

# Usage
input_file = 'devp.txt'  # Replace with your file path
tamil_output_file = 'tamil_sentences.txt'
english_output_file = 'english_sentences.txt'

split_sentences(input_file, tamil_output_file, english_output_file)
