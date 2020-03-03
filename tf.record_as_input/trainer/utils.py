"""
    Utilities
"""

def make_valid_dir(directory):
  """
      Make the given path a valid directory path
  """
  if directory[len(directory) - 1] != '/':
    directory += '/'
  return directory
