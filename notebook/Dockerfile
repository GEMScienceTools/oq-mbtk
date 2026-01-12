# Copyright (C) 2021-2025 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
#
#
#
ARG BASE_CONTAINER=python:3.12.6-slim
FROM $BASE_CONTAINER
LABEL maintainer="GEM Foundation <devops@openquake.org>" \
       vendor="GEM Foundation"
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -yq \
    wget git zip unzip \
    bzip2 \
    ca-certificates \
    gmt gmt-gshhg \
    ghostscript && \ 
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 -s /bin/bash openquake  
#
ARG oq_branch
ENV LANG=en_US.UTF-8
# Create venv
ENV GMT_SESSION_NAME=$$
#
# Install engine and tools
RUN pip install pip wheel setuptools --upgrade && \
    pip install notebook jupyterlab ipywidgets 
RUN cd /home/openquake && git clone -q --depth=1 -b $oq_branch https://github.com/gem/oq-engine.git && cd oq-engine && \
    pip --disable-pip-version-check install -e . -r requirements-py312-linux64.txt --no-warn-script-location 
RUN cd /home/openquake && git clone -q https://github.com/GEMScienceTools/oq-mbtk.git --depth=1 && cd oq-mbtk && \
    pip install -e . -r requirements-py312-linux.txt --no-warn-script-location 
RUN cd /home/openquake && git clone -q https://github.com/GEMScienceTools/hamlet.git --depth=1 && cd hamlet && \
    pip install -e . -r requirements.txt --no-warn-script-location 
#
#ENV OQ_CONFIG_FILE /opt/openquake/openquake.cfg
#ADD openquake.cfg /opt/openquake/openquake.cfg
#
USER openquake
WORKDIR /home/openquake
EXPOSE 8888
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser"]
