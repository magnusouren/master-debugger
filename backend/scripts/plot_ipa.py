import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

HORIZON_SECONDS = 10   # juster til din setup

file = sys.argv[1]
df = pd.read_csv(file)

obs_x=[]
obs_y=[]
pred_x=[]
pred_y=[]

for _,row in df.iterrows():

    if row["event_type"] not in {
        "user_state_estimate_logged",
        "observer_user_state_estimate_logged"
    }:
        continue

    try:
        payload=json.loads(row["data"])
    except:
        continue

    ipa = payload.get("contributing_features",{}).get("ipa_raw")

    if ipa is None:
        continue

    t=row["seconds_since_start"]

    if payload.get("is_predicted"):

        pred_x.append(t + HORIZON_SECONDS)
        pred_y.append(ipa)

    else:

        obs_x.append(t)
        obs_y.append(ipa)

plt.figure(figsize=(12,6))

plt.plot(obs_x,obs_y,label="Observed IPA")
plt.plot(pred_x,pred_y,label="Predicted IPA (aligned)")

plt.legend()
plt.grid()
plt.xlabel("Seconds")
plt.ylabel("IPA")

plt.show()