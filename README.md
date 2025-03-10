# NAD-CD (NIDS Alert Data with Concept Drift)

## Introduction

This repository hosts the NAD-CD data set for concept drift research in network IDS (NIDS) alert data. The data set was collected during 18 months (October 1 2022 -- March 31 2024) in TalTech Security Operations Center (https://doi.org/10.1109/CSR54599.2022.9850281) and contains over 11 million data points. For generating the data points of NAD-CD, the CSCAS stream clustering algorithm (https://doi.org/10.1016/j.future.2024.01.032) was used in real time. Each data point represents a group of one or more NIDS alerts which were raised for the same external IP address and for the same signature in a short time frame (the maximum size of the time frame was 5 minutes).

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

## Files in this repository

nad-cd.csv.bz2 -- NAD-CD data set

LICENSE -- license

drift-script.py -- implementation of 'ros', 'rus', 'al-trad', 'al-outlier', and 'random-outlier' methods for concept drift research. The script takes three parameters -- data directory, the method, and the model update strategy ('static', 'full', or 'cumulative'). Files in data directory have to conform with the following naming scheme -- dataset-YYYY-MM.csv or dataset-YYYY-MM-DD.csv. Each file from data directory must contain the data for the given month or day, and must begin with the same CSV header line as nad-cd.csv.

## Licensing

NAD-CD (NIDS Alert Data with Concept Drift), Copyright (C) 2025 Risto Vaarandi

NAD-CD is licensed under a Creative Commons Attribution 4.0 International License.

You should have received a copy of the license along with this work. If not, see https://creativecommons.org/licenses/by/4.0/.

## Author

Risto Vaarandi (firstname d0t lastname at gmail d0t c0m)
