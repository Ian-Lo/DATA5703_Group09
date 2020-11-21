# Image2Caption


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/xden2331/img2caption)

Image2Caption is a web application for captioning one image.

# New Features!

  - Upload one image and get the caption automatically!

### Tech

Dillinger uses a number of open source projects to work properly:

* [PyTorch] - The Deep Learning Lib.
* [Django] - Awesome web framework

### Installation

Clone repository

```sh
$ git clone https://github.com/xden2331/img2caption.git
```

Go to the work directory
```sh
$ cd img2caption
```

Install virtual environment
```sh
$ pipenv install
```

Download the caption model
```sh
$ chmod a+x main/download_caption_model.sh
$ ./main/download_caption_model.sh
```

Install the dependencies and devDependencies and start the server.

```sh
$ pipenv install -r requirements.txt
```

Start Django Server
```sh
$ python manage.py runserver
```

### Todos

 - Integrate more image captioning models
 - Deploy the web app

### Acknowledge

MexsonFernandes - The whole project is modified on [MexsonFernandes's project](https://github.com/MexsonFernandes/Img2VecCosSim-Django-Pytorch).
David Chung - The idea of this project is given by David Chung.
