rm -rf build dist graphagallery.egg-info
python setup.py check
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
