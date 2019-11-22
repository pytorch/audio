FROM ubuntu:latest

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

RUN apt update
RUN apt install -y sox libsox-dev libsox-fmt-all
RUN conda create -y --name python2.7 python=2.7
RUN conda create -y --name python3.5 python=3.5
RUN conda create -y --name python3.6 python=3.6
RUN conda create -y --name python3.7 python=3.7
SHELL [ "/bin/bash", "-c" ]
RUN source /usr/local/etc/profile.d/conda.sh && conda activate python2.7 && conda install -y -c conda-forge sox future
RUN source /usr/local/etc/profile.d/conda.sh && conda activate python3.5 && conda install -y -c conda-forge sox six
RUN source /usr/local/etc/profile.d/conda.sh && conda activate python3.6 && conda install -y -c conda-forge sox
RUN source /usr/local/etc/profile.d/conda.sh && conda activate python3.7 && conda install -y -c conda-forge sox
RUN echo "source /usr/local/etc/profile.d/conda.sh" >> ~/.bashrc
CMD [ "/bin/bash"]
