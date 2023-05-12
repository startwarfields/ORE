FROM fedora


RUN dnf -y update && dnf clean all
RUN dnf -y install python3 python3-pip && dnf clean all

ADD . /src

RUN cd /src; pip install -r requirements.txt

EXPOSE 8080

CMD python .ore_ui.py
