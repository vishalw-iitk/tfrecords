import apache_beam as beam

class DoFnMethods(beam.DoFn):
  def __init__(self):
    print('init---111')
    self.window = beam.window.GlobalWindow()
    #print(self.window)
  
  def setup(self):
    print('setup--111')

  def start_bundle(self):
    print('start_bundle----111')

  def process(self, element, window=beam.DoFn.WindowParam):
    print('***111****')
    self.window = window
    print('* process: ,' + element)
    yield '* process: ,' + element

  def finish_bundle(self):
    print("finish--1111")

  def teardown(self):
    print('teardown--1111')

class SplitWords(beam.DoFn):
  def __init__(self, delimiter):
    self.delimiter = delimiter
    print('init--222')

  def setup(self):
    print('setup---222')

  def start_bundle(self):
    print('start_bundle---222')

  def process(self, text):
    print('***222****')
    for word in text.split(self.delimiter):
      #print(text.split(self.delimiter))
      print(word)

  def finish_bundle(self):
    print("finish---222")

  def teardown(self):
    print('teardown')


with beam.Pipeline() as pipeline:

    plants = (
      pipeline
      | 'Gardening plants' >> beam.Create([
          '🍓Strawberry,🥕Carrot,🍆Eggplant',
          '🍅Tomato,🥔Potato',
      ])
      | 'DoFn methods' >> beam.ParDo(DoFnMethods())  ####1111
      | 'Split words' >> beam.ParDo(SplitWords(',')) ###222
      #| beam.Map(print)
  )