FROM python:buster

RUN apt-get update && apt-get install -y \
    texlive-lang-japanese \
    texlive-xetex \
    pandoc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

RUN pip install jupyter

RUN sed -i  /usr/local/share/jupyter/nbconvert/templates/latex/index.tex.j2 -e "s/\[11pt\]{article}/\[xelatex,ja=standard\]{bxjsarticle}/g"