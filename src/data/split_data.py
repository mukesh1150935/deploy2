import random

def split_data(input_file, train_file, test_file, dev_file, train_ratio=0.7, test_ratio=0.2, dev_ratio=0.1):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    random.shuffle(lines)
    
    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    test_end = train_end + int(total_lines * test_ratio)
    
    train_lines = lines[:train_end]
    test_lines = lines[train_end:test_end]
    dev_lines = lines[test_end:]
    
    with open(train_file, 'w', encoding='utf-8') as train_out:
        train_out.writelines(train_lines)
    
    with open(test_file, 'w', encoding='utf-8') as test_out:
        test_out.writelines(test_lines)
    
    with open(dev_file, 'w', encoding='utf-8') as dev_out:
        dev_out.writelines(dev_lines)
    
    print(f"Data split complete. {len(train_lines)} lines in {train_file}, {len(test_lines)} lines in {test_file}, {len(dev_lines)} lines in {dev_file}.")

def run():
    input_file = './data/combine.txt'
    train_file = './data/trainp.txt'
    test_file = './data/testp.txt'
    dev_file = './data/devp.txt'
    
    split_data(input_file, train_file, test_file, dev_file)


if __name__ == '__main__':
    run()
    
