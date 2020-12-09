# Some Notes
- Do not add tripinfo.xml to  your commits ever, it'll show up in the scenarios you work in, but we don't need to be
  sharing those files

# How to start virtual environment

Note: You will need 3.7 and PIP

If you have python 3.8 or greater installed, uninstall it.

first:

```pip install virtualenv```

then

Unix:
```virtualenv venv``` 

Windows:
```python -m venv venv``` or ```python3 -m venv venv```  if that fails.

depending on how you have python installed

To activate your virtual environment:

Unix:
```source ./venv/bin/activate```

Windows
```.\venv\Scripts\activate.bat```

Then
```pip install -r venv\requirements.txt```

To deactivate your virtual environment:

``deactivate``
