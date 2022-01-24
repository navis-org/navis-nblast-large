from setuptools import setup, find_packages
from collections import defaultdict
from typing import List, DefaultDict

import re

VERSIONFILE = "nblast/__version__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


extras_require: DefaultDict[str, List[str]] = defaultdict(list)
install_requires: List[str] = []
reqs = install_requires

with open("requirements.txt") as f:
    for line in f:
        if line.startswith("#extra: "):
            extra = line[8:].split("#")[0].strip()
            reqs = extras_require[extra]
        elif not line.startswith("#") and line.strip():
            reqs.append(line.strip())

dev_only = ["test-notebook", "dev"]
specialized = ['r']
all_dev_deps = []
all_deps = []
for k, v in extras_require.items():
    if k in specialized:
        continue
    all_dev_deps.extend(v)
    if k not in dev_only:
        all_deps.extend(v)

extras_require["all"] = all_deps
extras_require["all-dev"] = all_dev_deps

with open("README.md") as f:
    long_description = f.read()

setup(
    name='nblast',
    version=verstr,
    packages=find_packages(include=["nblast", "nblast.*"]),
    license='GNU GPL V3',
    description='NBLAST implementation optimized for large data sets',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/navis-org/navis-nblast-large',
    project_urls={
     "Documentation": "https://github.com/navis-org/navis-nblast-large",
     "Source": "https://github.com/navis-org/navis-nblast-large",
    },
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='Neuron Analysis Anatomy navis',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=install_requires,
    extras_require=dict(extras_require),
    tests_require=extras_require["dev"],
    # CI runs against >=3.7
    # but R-Python interface ships with 3.6 so this is necessary
    python_requires='>=3.6',
    zip_safe=False,

    include_package_data=True

)
