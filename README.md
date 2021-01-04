## Common Crawl processing tools

The code in this directory is part of an effort to make Web indexing available to all. We provide tools that allow a user to process the Common Crawl dumps ([https://commoncrawl.org](https://commoncrawl.org), containing petabytes of Web data in 40 languages).


### Getting raw text out of Common Crawl dumps

We will be using the .wet files from Common Crawl. For more information on the WET format, please consult [...].

Note that processing Common Crawl files is a very intensive job. Please refer to the information we have compiled about benchmarking (here in the wiki) before launching your own jobs. At the same time, don't be shy: you can process small amounts of data on your laptop without problems. So give it a go, and find friends to collectively process *more* data!



### Hash URLs in processed WET files

Each processed file now contains a number of raw text documents, which we need to further process to get a rough idea of the content of the page. In this step, we will take each raw text file, cluster the documents it contains using a technique called Locality Sensitive Hashing (LSH), and return some keywords representative of its content.





### Create a database for each hash

TODO
