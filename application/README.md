# Application PoC

The proof of concept application can be run with docker or directly with flask.<br>
Download [the trained models](https://drive.google.com/file/d/1z6koUTobj4kXH8LI1iXYmGAwPjMx-F_y/) and unzip to ```application/public/```.<br>
To run it as a docker container run the following from the ```/application``` directory:
```bash
docker build . -t mammography-project && docker run -p 8080:8080 -v ./public/assets/examples:/usr/app/public/assets/examples  mammography-project:latest
```

Open http://localhost:8080 and upload a mammogram. 