import os.path as op

from setuptools import setup, find_packages


# get the version (don't import mne here, so dependencies are not needed)
version = '0.0.0.0.1a'


setup(name='mvtoolbox',
      version=version,
      description='toolboc for multivariate data analysis',
      url='https://pyriemann.readthedocs.io',
      author='Goofy',
      author_email='gabriel@bccn-berlin.de',
      license='BSD (3-clause)',
      packages=[''],
      long_description_content_type='text/markdown',
      project_urls={
          'Documentation': 'https://pyriemann.readthedocs.io',
          'Source': 'https://github.com/pyRiemann/pyRiemann',
          'Tracker': 'https://github.com/pyRiemann/pyRiemann/issues/',
      },
      platforms='any',
      python_requires=">=3.6",
      install_requires=['numpy', 'scipy', 'scikit-learn',  'joblib', 'pandas'],
      extras_require={'docs': ['sphinx-gallery', 'sphinx-bootstrap_theme', 'numpydoc', 'mne', 'seaborn'],
                      'tests': ['pytest', 'seaborn', 'flake8']},
      zip_safe=False,
)
