# NAD-CD (NIDS Alert Data with Concept Drift)

## Introduction

This repository hosts the NAD-CD data set which can been created for researching concept drift in network IDS (NIDS) alert data. The data set was collected during 18 months (October 1 2022 -- March 31 2024) in TalTech Security Operations Center (https://doi.org/10.1109/CSR54599.2022.9850281) and contains over 11 million data points. Each data point represents a group of one or more NIDS alerts generated for the same external IP address and for the same signature in a short time frame (the maximum size of the time frame is 5 minutes). Since data points were produced in real time by the CSCAS stream clustering algorithm, each data point has a feature which indicates whether the point is an outlier.

## Data set fields

UNIXtime -- data point generation time (seconds since Epoch)

Timestamp -- data point generation time (textual timestamp)

SignatureText -- human readable alert text

SignatureID -- numeric signature ID

SignatureMatchesPerDay -- average number of matches per day by the signature that produced the current alert group (set to 0 if the signature has produced its first match less than 24 hours ago)

AlertCount -- the number of alerts in the current alert group

Proto -- numerical protocol ID (e.g., 6 denotes TCP and 17 UDP)

ExtIP -- anonymized IP address of the external host in integer format (i.e., IP address A.B.C.D is converted to A∗256^3 + B∗256^2 + C∗256 + D)

ExtPort -- port at the external host, set to -1 if alerts involved multiple external ports

IntIP -- anonymized IP address of the internal host in integer format, set to -1 if alerts involved multiple internal IP addresses

IntPort -- port at the internal host, set to -1 if alerts involved multiple internal ports

Similarity -- overall similarity with other alert groups from the same cluster (or with other outlier alert groups if the current alert group is an outlier), ranges from 0 to 1 (values close to 1 denote a high degree of similarity)

Label -- ground-truth label assigned by a human (0 denotes irrelevant and 1 important)

SCAS -- outlier indicator set by CSCAS (0 denotes inlier and 1 outlier)

AttrSimilarity -- similarity for the network IDS alert attribute Attr (there are 34 attributes in total). Set to -1 if the attribute Attr is not set for the given signature, otherwise ranges from 0 to 1. The field indicates how often has the attribute value been observed in other alert groups from the same cluster (or in other outlier alert groups if the current alert group is an outlier)

## Licensing

NIDS Alert Group Data Set, Copyright (C) 2025 Risto Vaarandi

NIDS Alert Group Data Set is licensed under a Creative Commons Attribution 4.0 International License.

You should have received a copy of the license along with this work. If not, see https://creativecommons.org/licenses/by/4.0/.

## Author

Risto Vaarandi (firstname d0t lastname at gmail d0t c0m)
