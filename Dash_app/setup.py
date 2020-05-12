from setuptools import setup


setup(
   name='Soft Robotics Materials Database',
   version='1.0',
   description='A useful module',
   author='Luc Marechal',
   author_email='',
   packages=['Soft Robotics Materials Database'],  #same as name
   install_requires=[
      'dash',
      'dash-html-components',
      'dash-bootstrap-components'], #external packages as dependencies
)
