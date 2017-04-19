#!/usr/bin/env python
# encoding: utf-8

import os
import sqlite3


class DBHandler(object):
    def __init__(self, fname):
        self.dbfile = fname
        if not os.path.exists(fname):
            raise IOError("Failed to provide a valid file")

    def yield_sentences(self, start, end):
        con = sqlite3.connect(self.dbfile)
        cur = con.cursor()
        try:
            if not start:
                cur.execute('select post_title, post_content from articles where post_publish_time <= "{} 23:59:59"'.format(end))
            else:
                cur.execute('select post_title, post_content from articles where post_publish_time >= "{} 00:00:00" and post_publish_time <= "{} 23:59:59"'.format(start, end))
            result = cur.fetchmany(size=1000)
            while result:
                for row in result:
                    title = row[0].strip()
                    content = row[1].strip()
                    if title:
                        yield title
                    if content:
                        yield content
                result = cur.fetchmany(size=1000)

            if not start:
                cur.execute('select reply_text from replies where reply_publish_time <= "{} 23:59:59"'.format(end))
            else:
                cur.execute('select reply_text from replies where reply_publish_time >= "{} 00:00:00" and reply_publish_time <= "{} 23:59:59"'.format(start, end))
            result = cur.fetchmany(size=1000)
            while result:
                for row in result:
                    content = row[0].strip()
                    if content:
                        yield content
                result = cur.fetchmany(size=1000)

            cur.close()
            con.close()
        except sqlite3.OperationalError, e:
            if e.message == 'no such table: replies':
                cur.close()
                con.close()
            else:
                raise e


if __name__ == '__main__':
    import codecs
    dh = DBHandler('./data/000001.sz.db')
    with codecs.open("./data/000001.sentences", 'w', 'utf-8') as fp:
        for s in dh.yield_sentences(None, '2017-03-30'):
            fp.write(s)
            fp.write("\n")
