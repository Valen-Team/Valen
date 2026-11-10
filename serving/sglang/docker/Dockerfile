FROM docker.io/lmsysorg/sglang:v0.5.23

ENV PYTHONUNBUFFERED=1 SGLANG_PLUGINS=valen
COPY serving/sglang /opt/valen/plugin
RUN python3 -m pip install /opt/valen/plugin

WORKDIR /opt/valen
EXPOSE 30000
ENTRYPOINT ["python3", "-m", "valen_sglang.launch_server", "--host", "0.0.0.0", "--attention-backend", "fa3"]
CMD []
