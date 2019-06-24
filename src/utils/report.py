#!/usr/bin/env python
# coding=utf-8
# import urllib2
from urllib import request
import json
import os
import time
import sys

URL = "/api/v1/reportmetrics"
HOST = "http://prajna-maintenance.wsd.com"


def reportmetrics(data):
    if sys.platform == "win32":
        return
    data = {
        "sAppinstanceName": os.environ.get("PRAJNA_APP_INST_ID", ""),
        "data": data,
    }
    full_url = HOST + URL
    json_data = json.dumps(data)
    req = request.Request(full_url)
    res = json.loads(request.urlopen(req, data=json_data.encode()).read())
    return res

if __name__ == "__main__":
    print(os.environ)
    for i in range(10):
        value = 0.001*i 
        data = [{"sMetricsName": "auc", "sMetricsValue": value}] 
        print(reportmetrics(data))
        time.sleep(10)
