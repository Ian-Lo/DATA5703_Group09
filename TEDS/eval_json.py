# load libraries
import os
import jsonlines
import sys
from datetime import datetime
from TEDS.metric import TEDS

cpus = os.cpu_count()
teds_metric = TEDS(n_jobs = cpus)


def create_folder(path):
    if os.path.exists(path) is False:
        os.mkdir(path)
        print(f"creating folder'{path}''")
    else:
        print(f"'{path}' exists")


def dl_by_id(file_id, filename, path, my_drive):
    target = path + filename
    file = my_drive.CreateFile({'id': file_id})
    file.GetContentFile(target)
    print(f"filename saved to {target}")

# Download file by name in Googledrive
# NOTE this may cause unexpected results if there are >1 files with the same name in google drive


def dl_by_name(filename, path, my_drive):
    target = path + filename
    # print(target)
    if os.path.exists(target) is True:
        print(f"'{target}' already exists")
    else:
        file_list = my_drive.ListFile(
            {'q': f'title="{filename}" and trashed = false'}).GetList()
        print(f'{len(file_list)} file(s) matching this name on Google Drive')
        file_id = file_list[0]['id']
        dl_by_id(file_id, filename, path, my_drive)


def build_html_structure(structure_information):
    ''' Build the structure skeleton of the HTML code
        Add the structural <thead> and the structural <tbody> sections to a fixed <html> header
    '''

    html_structure = '''<html>
                       <head>
                       <meta charset="UTF-8">
                       <style>
                       table, th, td {
                         border: 1px solid black;
                         font-size: 10px;
                       }
                       </style>
                       </head>
                       <body>
                       <table frame="hsides" rules="groups" width="100%%">
                         %s
                       </table>
                       </body>
                       </html>''' % ''.join(structure_information)

    return html_structure


def fill_html_structure(html_structure, cells_information):
    ''' Fill the structure skeleton of the HTML code with the cells content
        Every cell description is stored in a separate "token" field
        An initial assessment is performed to check that the cells content
        is compatible with the HTML structure skeleton
    '''

    import re
    from bs4 import BeautifulSoup as bs

    # initial compatibility assessment
    cell_nodes = list(re.finditer(r'(<td[^<>]*>)(</td>)', html_structure))
    assert len(cell_nodes) == len(cells_information), 'Number of cells defined in tags does not match the length of cells'

    # create a list with each cell content compacted into a single string
    cells = [''.join(cell['tokens']) for cell in cells_information]

    # sequentially fill the HTML structure with the cells content at the appropriate spots
    offset = 0
    html_string = html_structure
    for n, cell in zip(cell_nodes, cells):
        html_string = html_string[:n.end(1) + offset] + cell + html_string[n.start(2) + offset:]
        offset += len(cell)

    soup = bs(html_string, features="lxml") 
    html_string = soup.prettify(formatter='minimal')
    return html_string


# Take filename HTML string as input and test against GT JSON
def test_pred_html(img_name, pred_html, gt_file, max_count = 600000):
    # print(os.getcwd())
    reader = jsonlines.open(f'{gt_file}', 'r')
    count = 0

    # Loop through GT file 
    while count < max_count:
        count += 1
        if count % 10000 == 1:
            print(f'test_pred_html() count: {count}')
        try:
            annotation = next(reader.iter())
        except StopIteration:
            print("Oops!", sys.exc_info()[0], "occurred.")
            break
        except:
            print(
                f"{sys.exc_info()[0]} \
                last file processed {annotation['filename']}"
                )

        # Check for match of gt filename with img name
        if annotation['filename'] == img_name:
            img_filename = annotation['filename']
            img_struct = annotation['html']['structure']['tokens']
            img_cell = annotation['html']['cells']

            # Create valid HTML from structural tokens
            html_structure = build_html_structure(img_struct)
            # Merge structural and cell tokens
            true_html = fill_html_structure(html_structure, img_cell)

            # Test current prediction against Ground Truth
            
            test_pred_score, delta_t = teds_metric.evaluate(pred_html, true_html)
            # print(f'test_pred_html() score = {test_pred_score}')
            break

    print(f"test_pred_html() exit count: {count}")
    print(f'{img_filename}: {str(delta_t)} sec(s)')

    return test_pred_score, delta_t

# Takes pred and GT JSON and formats them into two dictionarys of PRED and GT
def json2html_TEDS(pred_jsonl, gt_jsonl, max_count):
    import sys

    reader = jsonlines.open(f'{pred_jsonl}', 'r') # Load JSON with predictions
    pred_html = {}
    gt_html = {}

    # Loop through predictions and generate valid HTML from structural and cell tokens
    count = 0
    while count < max_count:
        count += 1
        if count % 10000 == 1:
            print(f'PRED Cell count: {count}')
        try:
            annotation = next(reader.iter())
        except StopIteration:
            print("Oops!", sys.exc_info()[0], "occurred.")
            break
        except:
            print(f"{sys.exc_info()[0]} last file processed {annotation['filename']}")

        if annotation['filename']:
            img_filename = annotation['filename']
            img_struct = annotation['html']['structure']['tokens']
            img_cell = annotation['html']['cells']
            
            html_structure = build_html_structure(img_struct) # Create valid HTML from structural tokens
            html_string = fill_html_structure(html_structure, img_cell) # Merge structural and cell tokens
            
            pred_html[img_filename] = html_string # Create dictionary with fully formed HTML


    # Check if prediction is in GT and then generate valid HTML from structural/cell tokens
    reader = jsonlines.open(f'{gt_jsonl}', 'r') # Load JSON with Ground Truth
    count = 0 # Reset counter for GT loop
    match_count = 0 # count number of matches
    pred_img_fns = pred_html.keys()
    pred_img_fns_count = len(pred_img_fns)

    # Loop through GT file
    with jsonlines.open(f'{gt_jsonl}', 'r') as reader:
        while match_count < pred_img_fns_count: # Stop loop when count == number of PRED keys 
            if match_count % 10000 == 1:
                print(f'GT Cell count: {match_count}')
            match = False # reset exit condition
            while not match:
                try:
                    annotation = next(reader.iter())
                except StopIteration:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    break
                except:
                    print(f"{sys.exc_info()[0]} last file processed {annotation['filename']}")

                if annotation['filename'] in pred_img_fns: # Check PRED img in GT
                    img_filename = annotation['filename']
                    img_struct = annotation['html']['structure']['tokens']
                    img_cell = annotation['html']['cells']

                    html_structure = build_html_structure(img_struct) # Create valid HTML from structural tokens
                    html_string = fill_html_structure(html_structure, img_cell) # Merge structural and cell tokens
                    gt_html[img_filename] = html_string # Add HTML to Dictionary
                    match = True # Inner Exit condition
                    match_count += 1 # Increment outer exit condition
                    print(f'{img_filename} found in GT')
    print(f"number of CPUs detected = {cpus}\n #GTs == #PREDs {match_count == pred_img_fns_count}")
    return pred_html, gt_html

# Take PRED and GT JSONL to calculate TEDS score
# Pass JSON files in the same format as PubTabNet v2.0
def teds_jsonl_parallel(pred_jsonl, gt_jsonl, max_count = 600000):
    start_t = datetime.now()
    print(f'START: {start_t}')
    pred_html, gt_html = json2html_TEDS(pred_jsonl, gt_jsonl, max_count)
    pred_img_fns = pred_html.keys()
    pred_img_fns_count = len(pred_img_fns)

    # Parallel Eval PRED and GT HTML     
    from TEDS.parallel import parallel_process
    if pred_img_fns == 1:
        print(f"Only a single predicton {pred_img_fns_count}")
    else:
        inputs = [
                    {'pred':pred_html[fn], 'true':gt_html[fn]} 
                    for fn in pred_img_fns
                 ]        
        scores, delta_ts = parallel_process(
                                inputs, 
                                teds_metric.evaluate, # Function to parallelise
                                use_kwargs=True, 
                                n_jobs=cpus, # Number of threads to use
                                front_num=0 # First few jobs can be serialised to catch errors
                                )

    end_t = datetime.now()
    print(f"\n \
            \n\tSTART: {start_t} \
            \n\tEND: {end_t} \
            \n\tDELTA: {(str(end_t - start_t))} \
            ")
    # return_dict = {'TEDS_score':pred_score, 'pred_file':pred_jsonl}
    return dict(zip(pred_img_fns, scores, delta_ts)) #, pred_html, gt_html

# Single threaded for checking function performance
def teds_jsonl(pred_jsonl, gt_jsonl, max_count = 600000):
    start_t = datetime.now()
    print(f'START: {start_t}')
    pred_html, gt_html = json2html_TEDS(pred_jsonl, gt_jsonl, max_count)
    pred_img_fns = pred_html.keys()
    scores = []
    delta_ts = []

    # Eval PRED and GT HTML     
    for fn in pred_img_fns:
        
        score, delta_t = teds_metric.evaluate(pred_html[fn], gt_html[fn])
        scores.append(score)
        delta_ts.append(delta_t)

    end_t = datetime.now()
    print(f"\n \
            \n\tSTART: {start_t} \
            \n\tEND: {end_t} \
            \n\tDELTA: {(str(end_t - start_t))} \
            ")
    # return_dict = {'TEDS_score':pred_score, 'pred_file':pred_jsonl}
    return dict(zip(pred_img_fns, scores, delta_ts)) #, pred_html, gt_html




# write Scores to JSONL file
def TEDS_score2json(TEDS_score, output_path):
  scores = TEDS_score['TEDS_score'] # Dictionary
  write_fn = ([line for line in TEDS_score['pred_file'].split('/')
              if '.jsonl' in line
             ])[0] #ugly hack to turn list into string
  
  write_file = f"{write_fn[:-6]}-scores.jsonl"

  with jsonlines.open(output_path + write_file, mode='w') as writer:
    writer.write(scores)
    print(f'{write_file} written to colab {output_path}')
    return write_file, output_path


# write scores to folder in google drive
def w2gdrive_folder(filename, path, folder_id, my_drive):


    pydr_writer = my_drive.CreateFile({'parents': [{'id': folder_id}], 
                                       'title': filename})
    pydr_writer.SetContentFile(f'{path}{filename}')
    pydr_writer.Upload()
    print(f"{filename} uploaded")


def dl_by_listfile(list_file, path, my_drive):
    create_folder(path)
    for i in list_file:
        filename = i['title']
        file_id = i['id']
        target =path + filename
    
        if os.path.exists(target):
            print(f"'{target}' already exists")
        else:
            file = my_drive.CreateFile({'id': file_id})
            file.GetContentFile(target) 
            print(f"filename saved to {target}")
    return (path)

