FROM --platform=linux/amd64  pytorch/pytorch
RUN TARGETARCH=amd64 
RUN apt-get update && apt-get install -y libgl1 libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

# RUN apt-get install -y apt-transport-https
# RUN apt-get update -y apt-transport-https
# RUN apt-get install -y --no-install-recommends libgl1


# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN adduser --system --group user
USER user

WORKDIR /opt/app



COPY --chown=user:user requirements.txt /opt/app

RUN python -m pip install \
    --no-color \
    --requirement requirements.txt


COPY --chown=user:user helper.py /opt/app
COPY --chown=user:user inference.py /opt/app
COPY --chown=user:user weights/ /opt/algorithm/checkpoints/ 

ENTRYPOINT ["python", "inference.py"]
