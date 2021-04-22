# Copyright (C) 2021 GEM Foundation
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
ARG BASE_CONTAINER=jupyter/minimal-notebook
FROM $BASE_CONTAINER
LABEL maintainer="GEM Foundation <devops@openquake.org>" \
       vendor="GEM Foundation"
USER root

RUN apt-get update &&  \
    apt-get install -y gmt && \
    useradd -m -u 1001 -g 100 -s /bin/bash openquake 
ENV LANG en_US.UTF-8

USER openquake
ENV HOME /home/openquake
WORKDIR $HOME
RUN fix-permissions $HOME

# Install dependencies and engine
RUN git clone https://github.com/gem/oq-engine.git --depth=1 && cd oq-engine && \ 
    pip install  . -r requirements-py38-linux64.txt --no-warn-script-location && \ 
    cd $WORKDIR && \
    git clone https://github.com/GEMScienceTools/oq-mbtk.git --depth=1 && cd oq-mbtk && \
    pip install rtree && pip --disable-pip-version-check install . && \
    pip install --upgrade setuptools
#
#
