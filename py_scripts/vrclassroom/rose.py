# Read RotY FROM hmd_data, compute log histogram of 360 bins

import numpy as np
import math

def logged_histogram(data):
    """
	Count on 360 degress of RotY
	return ln(count+1)
    """
    result = [0] * 360
    for d in data:
        result[int(d)] += 1
    result = [math.log(r + 1) for r in result]
    return result


def main(hmd_data, CaseId):
    """
    in: single minimal hmd_data:[RotY,CaseId]

    out: [dic(CaseId,BlockNum, BinNum,Value)], len=360
    """

    rotYData = [r["RotY"] for r in hmd_data]
    his = logged_histogram(rotYData)
    result = []
    for j in range(360):
        result.append(dict(CaseId=CaseId, BlockNum=0, BinNum=j, Value=his[j]))
    return result
