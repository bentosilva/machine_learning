# encoding: utf-8
from bs4 import BeautifulSoup as BS

""" 这两个是试用文件，而且其中有一些新闻的 url 没有定义在 categories_2012.txt 中，后来由我手动添加 """
labelfile = 'categories_2012.txt'
newsfile = 'news_sohusite_xml.smarty.dat'


def load_labels():
    """
    parse labelfile, and return dict as url_prefix => category
    """
    labels = {}
    with open(labelfile, 'rb') as f:
        for line in f:
            line = line.strip().decode('gbk')
            label, url = line.split(u'：')
            labels[url] = label
    return labels


def load_news():
    """
    parse newsfile, and return dict as url => [docno, title, content]
    """
    news = {}
    # python2.7.6 下不能用 html_parser，报错 ... 用 html5lib 即可
    soup = BS(open(newsfile, 'rb'), 'html.parser', from_encoding='gb18030')
    # soup = BS(open(newsfile, 'rb'), 'html5lib')
    docs = soup.find_all('doc')
    for doc in docs:
        news[doc.url.text] = [doc.docno.text, doc.contenttitle.text, doc.content.text]
    return news


def find_label_by_url(url, labels):
    """
    Given url，iterate prefix in labels.keys()，and return label if match if found
    """
    for prefix, label in labels.iteritems():
        if url.find(prefix) == 0:
            return label
    return None


def numberize_labels(labels):
    """
    labels list to dic, which is like label_name -> label_id (string -> int)
    """
    uniq_labels = list(set(labels))
    dic = {}
    for i, label in enumerate(uniq_labels):
        dic[label] = i
    return dic, uniq_labels


def make_news_data():
    """
    iterate all loaded news，and return tags, text (title + content) in a lists respectively
    """
    labels = load_labels()
    news = load_news()
    texts = []
    tags = []
    for url, content in news.iteritems():
        texts.append(content[1] + content[2])
        tags.append(find_label_by_url(url, labels))
    dic, uniq_labels = numberize_labels(tags)
    tags = [dic[tag] for tag in tags]
    # 返回 uniq_labels 用于解码，从数值能够回推出 tag 名字
    return tags, texts, uniq_labels
