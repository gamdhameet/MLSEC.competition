FROM python:3.12-slim

#############################
# INSTALL PYTHON DEPENDENCIES
#############################

RUN apt-get -o Acquire::Max-FutureTime=100000 update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

# create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# copy and install python requirements
COPY docker-requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r docker-requirements.txt

#############################
# REBASE & DEPLOY CODE
#############################

FROM python:3.12-slim

# Copy python virtual env from previous image
COPY --from=0 /opt/venv /opt/venv

# Copy defender code and models
COPY defender /opt/defender/defender

#############################
# SETUP ENVIRONMENT
#############################

EXPOSE 8080

# Add defender user
RUN groupadd -r defender && useradd --no-log-init -r -g defender defender
USER defender

WORKDIR /opt/defender/

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/defender"
ENV DF_MODEL_DIR="/opt/defender/defender/models"
ENV DF_MODEL_THRESH="0.6"

#############################
# RUN CODE
#############################
CMD ["python", "-m", "defender"]