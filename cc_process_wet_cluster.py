"""Common Crawl WET processor - extraction for raw text data in a given language, from .wet files

Usage:
  cc_process_wet.py --file=<filename> --lang=<language>
  cc_process_wet.py (-h | --help)
  cc_process_wet.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --file=<filename>        Name of file with .wet file paths
  --lang=<language>        Language code

"""

import os
import sys
import gzip
import shutil
import requests
from time import sleep
from timer import Timer
from docopt import docopt
from warcio import ArchiveIterator
from langdetect import detect
from multiprocessing import Process
import os

num_cpu = 4

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def detect_language(text):
    language = "unk"
    try:
        language = detect(text)
        #print("Language:", language)
    except:
        pass
    return language


def read_doc_wet(record):
    url = None
    title = ""
    text = ""
    lang = "unk"

    try:
        url = record.rec_headers.get_header('WARC-Target-URI')
    except:
        pass

    if url:
        doc = record.content_stream().read()
        doc = doc.decode('utf-8')
        doc = doc.split('\n')
        title = doc[0]
        text = '\n'.join(doc[1:])
        lang = detect_language(title+' '+text[:1000])
    return url, title, text, lang



def write_from_wet(wet_url):
    print(wet_url)
    t = Timer()
    r = requests.get(wet_url, stream=True)
    records = ArchiveIterator(r.raw)

    n_documents=0

    file_path = wet_url.replace("https://","")
    file_path = file_path.replace('/','.')
    file_path = "./processed_wet/"+file_path.replace(".gz",".xml")
    f = open(file_path,'w')
    for i,record in enumerate(records):
        url, title, doc, lang = read_doc_wet(record)
        if not doc or not url or len(doc) < 1000:
            continue

        if record.rec_type == "conversion" and lang == 'en':
            f.write("<doc url="+url+" title="+title.replace(' ','_')+" lang="+lang+">"+'\n')
            f.write(doc+'\n')
            f.write("</doc>"+'\n')
            n_documents += 1
        if n_documents % 100 == 0:
            print(i,"documents processed",n_documents,"documents added...")
        sleep(0.01)
    f.close()
    return file_path

def process(jobs):
    info('function process')
    for wet_url in jobs:
        file_path = write_from_wet(wet_url)

        with open(file_path, 'rb') as f_in:
            with gzip.open(file_path+'.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)   

        os.unlink(file_path) 

def split_jobs(filename):
    urls = []
    f = open(filename,'r')
    for l in f:
        wet_url = "https://commoncrawl.s3.amazonaws.com/"+l.rstrip('\n')
        urls.append(wet_url)
    return list(urls[i::num_cpu] for i in range(num_cpu))



if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Processor 0.1')

    filename = args["--file"]
    jobs = split_jobs(filename)
    print(jobs[0])

    p0 = Process(target=process, args=(jobs[0],))
    p1 = Process(target=process, args=(jobs[1],))
    p2 = Process(target=process, args=(jobs[2],))
    p3 = Process(target=process, args=(jobs[3],))

    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p0.join()
    p1.join()
    p2.join()
    p3.join()
