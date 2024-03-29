**This code is undergoing updates. You can take a peak at your own risk :)**

# Common Crawl processing tools

The code in this directory is part of an effort to make Web indexing available to all. We provide tools that allow a user to process the Common Crawl dumps ([https://commoncrawl.org](https://commoncrawl.org), containing petabytes of Web data in 40 languages).


### Getting raw text out of Common Crawl dumps

We will be using the .wet files from Common Crawl. For more information on the WET format, please consult [...].

Note that processing Common Crawl files is a very intensive job. Please refer to the information we have compiled about benchmarking (here in the wiki) before launching your own jobs. At the same time, don't be shy: you can process small amounts of data on your laptop without problems. So give it a go, and find friends to collectively process *more* data!

Before you start, you will have to find the location of some .wet files to process. If you go to the Common Crawl website and look for monthly file listings, for instance [here](https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-50/index.html), you will find files named *wet.paths.gz*. If you uncompress one of those *wet.paths* file, you will get a list of URLs starting with *crawl-data...* Prepend *https://commoncrawl.s3.amazonaws.com/* to each line, and you will get a few tens of thousands of .wet files' URLs.


### Hash URLs in processed WET files

Each processed file now contains a number of raw text documents, which we need to further process to get a rough idea of the content of the page. In this step, we will take each raw text file, cluster the documents it contains using a technique called Locality Sensitive Hashing (LSH), and return some keywords representative of its content.


### Create a database for each hash

TODO


## Using the code

We recommend using a virtual environment. You can set it up from outside your clone repository, by doing:

     virtualenv common-crawl-processor

To process raw .wet files, do:

    python cc_process_wet.py --file=example-path-file.txt --lang=en
    
You should see the file being processed:

    1 documents processed 0 documents added...
    3 documents processed 0 documents added...
    383 documents processed 100 documents added...
    384 documents processed 100 documents added...
    385 documents processed 100 documents added...
    387 documents processed 100 documents added...
    768 documents processed 200 documents added...
