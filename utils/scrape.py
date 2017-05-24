# Scrape sonnets from sonnets.org...

import urllib2
from bs4 import BeautifulSoup
from timeit import default_timer as timer
import cPickle as pickle
import codecs
import json

from xml_ops import TreeBuilder

def get_sonnets(raw_html):
  raw_sonnets = raw_html.split('<h2>')[1:]
  sonnets = []
  for raw_sonnet in raw_sonnets:
    end_title = raw_sonnet.find('</h2>')
    title = raw_sonnet[:end_title].strip()
    dts = raw_sonnet.split('<dt>')[1:]
    raw_lines = [dt.split('<')[0].strip() for dt in dts]
    lines = [line for line in raw_lines if len(line) > 0]
    sonnets.append((title, '\n'.join(lines)))
  return sonnets


def main():
  request = urllib2.urlopen("http://sonnets.org/alpha.htm")
  index_html = request.read().decode('latin-1')
  index_soup = BeautifulSoup(index_html, "html5lib")
  all_pages = index_soup.find_all("a")
  pages = []
  for a in all_pages:
    if a.parent.name != "h3":
      continue
    author = a.get_text(strip=True)
    url = "http://sonnets.org/" + a["href"]
    pages.append((author, url))

  print "found %d links" % len(pages)

  start = timer()
  count = 0
  data = []

  f_out = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/raw_sonnets.txt"

  with codecs.open(f_out, "w+", encoding="utf-8") as f:
    for i,(author,url) in enumerate(pages):
      try:
        raw_html = urllib2.urlopen(url).read() #.decode('latin-1')
        decoded_html = BeautifulSoup(raw_html).prettify()
        sonnets = get_sonnets(decoded_html)
        data.append((author, sonnets))
        print "{}/{}".format(i, len(pages))
        count += 1

        if len(sonnets) > 0:
          f.write("<author>" + author + "\n\n")
          for title, sonnet in sonnets:
            f.write("<title>" + title + "\n\n")
            f.write("<sonnet>" + sonnet + "\n\n\n")


      except urllib2.HTTPError:
        print "Couldn't open {}".format(url)
      except Exception as e:
        print "Error: %s" % str(e)

  print "Processed {} sonnets in {}s".format(count, timer() - start)
  print "Pickling..."

  fn = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/sonnets.pickle"

  with open(fn, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

  print "Done."

def clean_line(line):
  i = len(line) - 1
  while i > 0 and line[i].isnumeric():
    i -= 1
  return line[:i+1].strip()

def clean_lines(raw_lines):
  lines = [line.strip() for line in raw_lines]
  poem = [clean_line(line) for line in lines if len(line) > 0]
  if len(poem) > 0:
    last = poem[-1][:-1].strip()
    if last.isnumeric():
      poem.pop()
  return poem


def go():
  f_out = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/scratch.txt"

  author = "Longfellow, Henry Wadsworth"

  count = 0
  start = timer()

  with codecs.open(f_out, "w", encoding="utf-8") as f:
    f.write("<author>" + author + "\n\n")
    for i in xrange(1, 617):
      msg = "{}/{}".format(i, 616)
      try:
        url = "http://www.bartleby.com/356/{}.html/".format(i)
        html = urllib2.urlopen(url).read()
        soup = BeautifulSoup(html, "html5lib")

        raw_poem = soup.find("pre").get_text(strip=True)

        raw_lines = raw_poem.split('\n')
        
        if len(raw_lines) < 20:
          lines = clean_lines(raw_lines)
          if len(lines) >= 14 and len(lines) <= 16:
            title = soup.find("h3").get_text(strip=True)
            sonnet = "\n".join(lines)
            f.write("<title>" + title + "\n\n")
            f.write("<sonnet>" + sonnet + "\n\n\n")
            count += 1
            msg += "*"

        print msg

      except urllib2.HTTPError:
        print "Couldn't open {}".format(url)
      except Exception as e:
        print "Error: %s" % str(e)

      # if i > 120:
      #   break

  print "found {} ~sonnets~ in {}s".format(count, timer() - start)




def quick_process():
  fn = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/scratch.txt"

  with codecs.open(fn, "r", encoding="utf-8") as f:
    text = f.read()

  with codecs.open(fn, "w", encoding="utf-8") as f:
    chunks = text.split('\n\n')
    data = []
    for i,chunk in enumerate(chunks):
      if i % 2 == 0:
        title = chunk.strip()
        f.write("<title>" + title + "\n\n")
      else:
        f.write("<sonnet>" + chunk + "\n\n\n")

def write_json():
  fn = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/sonnets.txt"

  with codecs.open(fn, "r", encoding="utf-8") as f:
    text = f.read()

  data = {}

  chunks = text.split("<author>")
  for chunk in chunks:
    sub_chunks = chunk.split("<title>")
    author = sub_chunks[0].strip()
    if author not in data:
      data[author] = []
    for sub_chunk in sub_chunks[1:]:
      try:
        raw_title, raw_sonnet = sub_chunk.split("<sonnet>")
        title = raw_title.strip()
        sonnet = raw_sonnet.strip()
        data[author].append((title, sonnet))
      except Exception as e:
        print sub_chunk
        print e


  f_out = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/sonnets.json"
  with codecs.open(f_out, "w", encoding="utf-8") as f:
    json.dump(data, f, sort_keys=True, indent=2,
              separators=(',', ': '), ensure_ascii=False)

def write_xml():
  f_in = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/sonnets.json"
  f_out = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/sonnets.xml"

  with codecs.open(f_in, "r", encoding="utf-8") as f:
    data = json.load(f, encoding="utf-8")

  tree = TreeBuilder()
  tree.open("data")
  for key in data:
    tree.open("page")
    tree.add_attr("author", key)
    tree.open("sonnets")
    for title,text in data[key]:
      tree.open("sonnet")
      tree.add_attr("title", title)
      tree.open("text")
      for line in text.split("\n"):
        tree.add_data(line)
      tree.close("text")
      tree.close("sonnet")
    tree.close("sonnets")
    tree.close("page")
  tree.close("data")

  with codecs.open(f_out, "w", encoding="utf-8") as f:
    f.write(tree.to_string())

def pre_extract(html):
  soup = BeautifulSoup(html, "html5lib")
  raw_poem = soup.find("pre").get_text(strip=True)
  raw_lines = raw_poem.split("\n")
  title, sonnet = None, None
  if len(raw_lines) < 20:
    lines = clean_lines(raw_lines)
    if len(lines) >= 14 and len(lines) <= 16:
      title = soup.find("h3").get_text(strip=True)
      sonnet = "\n".join(lines)
  return title, sonnet


def table_extract(html):
  start_title = html.find("<!-- BEGIN CHAPTERTITLE -->")
  end_title = html.find("<!-- END CHAPTERTITLE -->")
  title_soup = BeautifulSoup(html[start_title:end_title], "html5lib")
  title = title_soup.find_all("b")[-1]
  if title is not None:
    title = title.get_text(strip=True)

  sonnet = None
  start_ch = html.find("<!-- BEGIN CHAPTER -->")
  end_ch= html.find("<!-- END CHAPTER -->")
  soup = BeautifulSoup(html[start_ch:end_ch], "html5lib")
  if soup is not None:
    raw_lines = soup.get_text().split("\n")
    lines = clean_lines(raw_lines)
    if len(lines) >= 14 and len(lines) <= 16:
      sonnet = "\n".join(clean_lines(raw_lines))

  return title, sonnet



def scrape_sonnets(i_start, i_end, extract=table_extract):
  f_out = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/scratch.txt"

  base_url = "http://www.bartleby.com/358/{}.html"

  count = 0
  start = timer()

  with codecs.open(f_out, "w", encoding="utf-8") as f:
    for i in xrange(i_start, i_end+1):
      msg = "{}/{}".format(i, i_end)
      try:
        url = base_url.format(i)
        html = urllib2.urlopen(url).read()
        
        title, sonnet = extract(html)
        if sonnet is not None:
          f.write(title + "\n\n")
          f.write(sonnet + "\n\n\n")
          count += 1
          msg += "*"

        print msg

      except urllib2.HTTPError:
        print "Couldn't open {}".format(url)
      except Exception as e:
        print "Error: %s" % str(e)

      # if i > 120:
      #   break

  print "found {} ~sonnets~ in {}s".format(count, timer() - start)

def add_to_json(author):
  f_in = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/scratch.txt"
  f_out = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/sonnets.json"

  with codecs.open(f_in, "r", encoding="utf-8") as f:
    text = f.read()
    raw_sonnets = text.split("\n\n\n")
    sonnets = [tuple(raw_sonnet.split("\n\n")) for raw_sonnet in raw_sonnets]

  with codecs.open(f_out, "r", encoding="utf-8") as f:
    data = json.load(f, encoding="utf-8")

  if author not in data:
    data[author] = []

  data[author] += sonnets


  with codecs.open(f_out, "w", encoding="utf-8") as f:
    json.dump(data, f, sort_keys=True, indent=2,
              separators=(',', ': '), ensure_ascii=False)

  num_sonnets = sum([len(data[key]) for key in data])
  print "Added {} new sonnets; total sonnets: {}".format(len(sonnets), num_sonnets)

def write_sonnets():
  f_in = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/sonnets.json"
  f_out = "/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/tf/data/sonnets.txt"

  with codecs.open(f_in, "r", encoding="utf-8") as f:
    data = json.load(f, encoding="utf-8")

  with codecs.open(f_out, "w", encoding="utf-8") as f:
    for key in data:
      try:
        for _, sonnet in data[key]:
          f.write(sonnet + "\n\n")
      except Exception as e:
        print str(e)
        print "Error with " + key


if __name__ == "__main__":
  #scrape_sonnets(1011, 1132)
  #add_to_json("Linche, Richard")
  write_sonnets()
