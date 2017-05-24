# Provide functions for reading and writing XML documents

class TreeBuilder(object):
  def __init__(self, indent=2):
    self._tree = []
    self._indent = 2
    self._cur_indent = 0

  def open(self, tag):
    line = " " * self._cur_indent + "<" + tag + ">"
    self._tree.append(line)
    self._cur_indent += self._indent

  def add_data(self, data):
    line = " " * self._cur_indent + data
    self._tree.append(line)

  def close(self, tag):
    if self._cur_indent - self._indent < 0:
      raise ValueError("No tag to close")
    self._cur_indent -= self._indent
    line = " " * self._cur_indent + "</" + tag + ">"
    self._tree.append(line)

  def add_attr(self, tag, attr):
    self.open(tag)
    self.add_data(attr)
    self.close(tag)

  def to_string(self):
    return "\n".join(self._tree)
