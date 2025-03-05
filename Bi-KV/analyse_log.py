import argparse


def find_logs_with_req_id(log_file_path, req_id, output_file_path):
    try:
        with open(log_file_path, 'r',errors='ignore') as infile, open(output_file_path, 'w') as outfile:
            for line in infile:
                if f'req_id={req_id}' in line:
                    outfile.write(line)
    except FileNotFoundError:
        print(f"The file {log_file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def find_all_process_req_ids(log_file_path, output_file_path):
    try:
        seen_ids = set()
        with open(log_file_path, 'r', encoding='utf-8', errors='replace') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if 'process req_id=' in line:
                    # start_index = line.find('process req_id=') + len('process req_id=')
                    # end_index = line.find(' ', start_index)
                    # current_id = line[start_index:end_index].strip()
                    # if current_id not in seen_ids:
                    #     seen_ids.add(current_id)
                    outfile.write(line)
                elif 'excute requests cost' in line:
                    outfile.write(line)
    except FileNotFoundError:
        print(f"The file {log_file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def find_all_put_req_ids(log_file_path, output_file_path):
    try:
        seen_ids = set()
        with open(log_file_path, 'r', encoding='utf-8', errors='replace') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if 'put req_id=' in line:
                    # start_index = line.find('process req_id=') + len('process req_id=')
                    # end_index = line.find(' ', start_index)
                    # current_id = line[start_index:end_index].strip()
                    # if current_id not in seen_ids:
                    #     seen_ids.add(current_id)
                    outfile.write(line)
                elif 'excute requests cost' in line:
                    outfile.write(line)
    except FileNotFoundError:
        print(f"The file {log_file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='Log Analyzer')
    parser.add_argument('f', help='function to run',type=int)

    args = parser.parse_args()

    log_file_path = 'debug250303-time37.log'
    log_name = log_file_path.split('.')[0]
    req_id = '128'
    output_file_path = f'filtered_{log_name}_1.log'
    
    if args.f == 1:
        find_logs_with_req_id(log_file_path, req_id, output_file_path)
    elif args.f == 2:
        find_all_process_req_ids(log_file_path, output_file_path)
    elif args.f == 3:
        find_all_put_req_ids(log_file_path, output_file_path)

# Example usage
if __name__ == '__main__':
    main()



