def count_lines(file_path):
    """Counts the number of lines in a file, excluding the header if present."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for line in file)

def create_settings_file(src_file, tgt_file, output_file):
    """Creates a settings file with SRC_VOCAB and TGT_VOCAB variables."""
    src_vocab = count_lines(src_file)
    tgt_vocab = count_lines(tgt_file)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"SRC_VOCAB = {src_vocab}\n")
        file.write(f"TGT_VOCAB = {tgt_vocab}\n")

    # print(f"Settings file '{output_file}' created successfully.")
    # print(f"SRC_VOCAB = {src_vocab}, TGT_VOCAB = {tgt_vocab}")

# File paths
eng_index_dict_file = './data/word_name_dict/en_index_dict.csv'
eng_word_dict_file = './data/word_name_dict/pu_index_dict.csv'
output_settings_file = 'setting2.py'




def run():
    # Create the settings file
    create_settings_file(eng_index_dict_file, eng_word_dict_file, output_settings_file)
    


if __name__ == "__main__":
    run()


# \src\data\word_name_dict\en_index_dict.csv