# EXPIL
The implementation of the NeSy-RL system EXPIL.


### Setup locally

``` 
pip install -r requirements.txt
``` 

### Setup remotely

build a docker

``` 
docker build -t expil_docker .
```

run the docker

``` 
docker run  --gpus all -it -v path/to/storage:/EXPIL/storage --rm expil_docker
```

### Experiments

See this [Readme](expil/README.md) in folder `EXPIL/expil`
