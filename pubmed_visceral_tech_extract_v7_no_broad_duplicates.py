#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PubMed extraction v7: visceral/abdominal surgery operations/diseases/techniques
AND concrete surgical technology / physics / device / software / material terms, 1980-2025.

Why v7?
- Uses modular subqueries plus automatic 10-window/year/month date splitting to avoid PubMed query-length and ~9999-ID retrieval problems.
- Every subquery requires at least one visceral surgery/procedure/disease term AND at least one technical/device/physics term.
- Exports a deduplicated Excel + CSV with metadata and which query modules matched each PMID.

Install:
    python -m pip install requests pandas openpyxl tqdm

Recommended environment variables:
    setx NCBI_EMAIL "your.email@example.com"
    setx NCBI_API_KEY "your_ncbi_api_key"

Run:
    python pubmed_visceral_tech_extract_v7_no_broad_duplicates.py
"""

from __future__ import annotations

import os
import re
import time
import json
import html
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DB = "pubmed"
EMAIL = os.getenv("NCBI_EMAIL", "your.email@example.com")
API_KEY = os.getenv("NCBI_API_KEY", "")
TOOL = "visceral_surgery_tech_bridge_extractor_v7_no_broad_duplicates"

SEARCH_RETMAX = 9999        # keep below PubMed large-query retrieval cap for ESearch IDs
DATE_SPLIT_THRESHOLD = 9999 # if a subquery has more hits, split by time windows
FETCH_BATCH_SIZE = 200      # EFetch XML batch size
START_DATE = "1980/01/01"
END_DATE = "2025/12/31"
OUTPUT_PREFIX = "pubmed_visceral_surgery_technology_precise_1980_2025_v7_no_broad_duplicates"
OUTPUT_XLSX = f"{OUTPUT_PREFIX}.xlsx"
OUTPUT_CSV = f"{OUTPUT_PREFIX}.csv"
OUTPUT_QUERY_TXT = f"{OUTPUT_PREFIX}_queries.txt"
OUTPUT_QUERY_JSON = f"{OUTPUT_PREFIX}_query_modules.json"

# -----------------------------------------------------------------------------
# PubMed query helpers
# -----------------------------------------------------------------------------

def tiab(term: str) -> str:
    """Quote term and add [Title/Abstract]."""
    term = term.strip()
    if term.endswith("[Mesh]") or term.endswith("[MeSH Terms]") or "[" in term:
        return term
    escaped = term.replace('"', '\\"')
    return f'"{escaped}"[Title/Abstract]'


def or_block(terms: Iterable[str]) -> str:
    """Build an OR block from terms."""
    return "(\n  " + "\n  OR ".join(tiab(t) for t in terms if str(t).strip()) + "\n)"


def clean_query(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()

DATE_ABSTRACT_HUMAN = clean_query(f'''
("{START_DATE}"[Date - Publication] : "{END_DATE}"[Date - Publication])
AND hasabstract[text]
NOT (animals[MeSH Terms] NOT humans[MeSH Terms])
''')


# Ten broad publication-date windows. Large windows are automatically split again by year/month.
DATE_WINDOWS_10 = [
    ("1980/01/01", "1984/12/31"),
    ("1985/01/01", "1989/12/31"),
    ("1990/01/01", "1994/12/31"),
    ("1995/01/01", "1999/12/31"),
    ("2000/01/01", "2004/12/31"),
    ("2005/01/01", "2009/12/31"),
    ("2010/01/01", "2014/12/31"),
    ("2015/01/01", "2019/12/31"),
    ("2020/01/01", "2022/12/31"),
    ("2023/01/01", "2025/12/31"),
]


def pub_date_filter(start_date: str, end_date: str) -> str:
    """PubMed date filter for one split window."""
    return clean_query(f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])')


def year_windows(start_year: int, end_year: int) -> List[Tuple[str, str]]:
    return [(f"{y}/01/01", f"{y}/12/31") for y in range(start_year, end_year + 1)]


def month_windows(year: int) -> List[Tuple[str, str]]:
    # PubMed accepts month/day ranges. Use 31 as the end day for all months; PubMed handles date ranges permissively.
    return [(f"{year}/{m:02d}/01", f"{year}/{m:02d}/31") for m in range(1, 13)]

SURGERY_CONTEXT_MESH = [
    # Clinical/surgical context only. Generic technology/procedure MeSH terms such as
    # Laparoscopy[Mesh], Robotic Surgical Procedures[Mesh], Endoscopy[Mesh] and Surgical Stapling[Mesh]
    # are deliberately NOT used here, because they create circular queries when paired with technology modules.
    "General Surgery[Mesh]", "Digestive System Surgical Procedures[Mesh]",
    "Gastrointestinal Surgical Procedures[Mesh]", "Anastomosis, Surgical[Mesh]",
    "Hepatectomy[Mesh]", "Pancreatectomy[Mesh]", "Colectomy[Mesh]", "Cholecystectomy[Mesh]",
    "Bariatric Surgery[Mesh]", "Hernia, Inguinal/surgery[Mesh]", "Hernia, Ventral/surgery[Mesh]",
    "Esophagectomy[Mesh]", "Gastrectomy[Mesh]", "Appendectomy[Mesh]"
]

# -----------------------------------------------------------------------------
# Visceral / abdominal surgery modules
# -----------------------------------------------------------------------------
SURGERY_MODULES: Dict[str, List[str]] = {
    "general_abdominal_surgery_context": SURGERY_CONTEXT_MESH + [
        # Generic clinical context only. Access/device words such as laparoscopy, trocar,
        # pneumoperitoneum, insufflation, SILS, NOTES etc. are kept exclusively in technology modules.
        "visceral surgery", "abdominal surgery", "digestive surgery", "gastrointestinal surgery", "GI surgery",
        "general surgery", "upper gastrointestinal surgery", "upper GI surgery", "lower gastrointestinal surgery",
        "abdominal operation", "abdominal procedure", "abdominal resection", "bowel surgery", "intestinal surgery",
        "foregut surgery", "hindgut surgery", "oncologic surgery", "emergency abdominal surgery",
        "laparotomy", "open abdomen", "damage control surgery", "enhanced recovery after surgery", "ERAS",
        "surgical reconstruction", "surgical repair", "surgical resection", "surgical complication",
        "postoperative complication", "intraoperative complication"
    ],
    "upper_gi_esophageal_gastric_bariatric": [
        "esophagectomy", "oesophagectomy", "Ivor Lewis", "McKeown", "transhiatal esophagectomy",
        "minimally invasive esophagectomy", "MIE", "thoracoscopic esophagectomy", "esophagogastric anastomosis",
        "gastric pull-up", "gastric conduit", "gastroesophageal junction", "GEJ", "achalasia", "Heller myotomy",
        "POEM", "peroral endoscopic myotomy", "fundoplication", "Nissen fundoplication", "Toupet fundoplication",
        "Dor fundoplication", "hiatal hernia repair", "paraesophageal hernia repair", "cruroplasty",
        "gastropexy", "LINX", "magnetic sphincter augmentation", "anti-reflux surgery", "GERD surgery",
        "gastrectomy", "total gastrectomy", "subtotal gastrectomy", "distal gastrectomy", "proximal gastrectomy",
        "pylorus-preserving gastrectomy", "wedge resection", "gastric wedge resection", "sleeve gastrectomy",
        "gastric bypass", "Roux-en-Y gastric bypass", "RYGB", "one anastomosis gastric bypass", "OAGB",
        "mini gastric bypass", "biliopancreatic diversion", "duodenal switch", "SADI-S", "bariatric surgery",
        "metabolic surgery", "gastrojejunostomy", "esophagojejunostomy", "Billroth I", "Billroth II",
        "Roux-en-Y reconstruction", "Braun anastomosis", "jejunojejunostomy", "D2 lymphadenectomy",
        "gastric cancer surgery", "gastrointestinal stromal tumor", "GIST", "submucosal tumor"
    ],
    "colorectal_anorectal_ibd": [
        "colorectal surgery", "colon surgery", "rectal surgery", "proctology", "colorectal resection",
        "colectomy", "right colectomy", "right hemicolectomy", "left hemicolectomy", "transverse colectomy",
        "sigmoidectomy", "proctectomy", "low anterior resection", "LAR", "anterior resection",
        "ultralow anterior resection", "abdominoperineal resection", "APR", "Hartmann procedure",
        "Hartmann reversal", "total mesorectal excision", "TME", "complete mesocolic excision", "CME",
        "D3 lymphadenectomy", "lateral lymph node dissection", "transanal total mesorectal excision", "TaTME",
        "transanal minimally invasive surgery", "TAMIS", "transanal endoscopic microsurgery", "TEM", "TEO",
        "intersphincteric resection", "ISR", "coloanal anastomosis", "colorectal anastomosis",
        "ileocolic anastomosis", "ileorectal anastomosis", "ileal pouch-anal anastomosis", "IPAA",
        "pouch surgery", "Kono-S", "strictureplasty", "ileocecal resection", "small bowel resection",
        "stoma", "ileostomy", "colostomy", "diverting ileostomy", "stoma closure", "stoma reversal",
        "rectopexy", "ventral mesh rectopexy", "hemorrhoidectomy", "stapled hemorrhoidopexy", "PPH",
        "fistula-in-ano", "anal fistula", "seton", "LIFT procedure", "VAAFT", "FiLaC",
        "pilonidal sinus", "Crohn surgery", "ulcerative colitis surgery", "IBD surgery",
        "diverticulitis surgery", "colon cancer", "rectal cancer", "colorectal cancer"
    ],
    "hpb_liver_biliary_gallbladder": [
        "hepatobiliary surgery", "HPB surgery", "liver surgery", "hepatic surgery", "hepatectomy",
        "liver resection", "hepatic resection", "minor hepatectomy", "major hepatectomy", "right hepatectomy",
        "left hepatectomy", "right posterior sectionectomy", "right anterior sectionectomy", "left lateral sectionectomy",
        "segmentectomy", "liver segmentectomy", "bisegmentectomy", "sectionectomy", "trisectionectomy",
        "trisegmentectomy", "lobectomy", "liver wedge resection", "non-anatomic resection", "anatomic resection",
        "parenchymal transection", "Pringle maneuver", "Glissonean approach", "Laennec approach",
        "hanging maneuver", "ALPPS", "associating liver partition", "portal vein embolization", "PVE",
        "two-stage hepatectomy", "liver metastases", "colorectal liver metastases", "CRLM",
        "hepatocellular carcinoma", "HCC", "cholangiocarcinoma", "intrahepatic cholangiocarcinoma",
        "perihilar cholangiocarcinoma", "Klatskin tumor", "gallbladder cancer", "biliary tract cancer",
        "cholecystectomy", "laparoscopic cholecystectomy", "single incision cholecystectomy", "subtotal cholecystectomy",
        "bile duct exploration", "common bile duct exploration", "choledochotomy", "choledochoscopy",
        "hepaticojejunostomy", "choledochojejunostomy", "bilioenteric anastomosis", "biliary reconstruction",
        "Roux-en-Y hepaticojejunostomy", "bile duct injury", "biliary stricture", "bile leak", "cholecystitis",
        "gallstone", "choledocholithiasis"
    ],
    "pancreatic_duodenal_spleen": [
        "pancreatic surgery", "pancreatectomy", "pancreaticoduodenectomy", "Whipple", "pylorus preserving pancreaticoduodenectomy",
        "PPPD", "distal pancreatectomy", "central pancreatectomy", "middle pancreatectomy", "total pancreatectomy",
        "pancreatic enucleation", "duodenum-preserving pancreatic head resection", "DPPHR", "Beger procedure",
        "Frey procedure", "Puestow procedure", "lateral pancreaticojejunostomy", "pancreaticojejunostomy",
        "pancreaticogastrostomy", "duct-to-mucosa", "Blumgart anastomosis", "invagination pancreaticojejunostomy",
        "pancreatic fistula", "postoperative pancreatic fistula", "POPF", "pancreatic leak", "pancreatic stump",
        "pancreatic cancer", "pancreatic ductal adenocarcinoma", "PDAC", "IPMN", "pancreatic neuroendocrine tumor",
        "duodenal surgery", "duodenal resection", "duodenojejunostomy", "splenectomy", "spleen preserving distal pancreatectomy"
    ],
    "hernia_abdominal_wall": [
        "hernia repair", "inguinal hernia repair", "femoral hernia repair", "ventral hernia repair", "incisional hernia repair",
        "umbilical hernia repair", "epigastric hernia repair", "parastomal hernia repair", "hiatal hernia repair",
        "diaphragmatic hernia repair", "abdominal wall reconstruction", "component separation", "posterior component separation",
        "anterior component separation", "transversus abdominis release", "TAR", "Rives-Stoppa", "Stoppa repair",
        "retromuscular repair", "sublay repair", "onlay repair", "IPOM", "intraperitoneal onlay mesh",
        "eTEP", "TEP", "TAPP", "Lichtenstein repair", "Shouldice repair", "Bassini repair", "plug and patch",
        "mesh repair", "mesh fixation", "tacker fixation", "fibrin glue fixation", "self-gripping mesh", "abdominal wall defect"
    ],
    "advanced_endoscopy_endoluminal": [
        "endoscopic submucosal dissection", "ESD", "endoscopic mucosal resection", "EMR", "endoscopic full-thickness resection",
        "EFTR", "full-thickness resection device", "FTRD", "submucosal tunneling endoscopic resection", "STER",
        "peroral endoscopic myotomy", "POEM", "gastric peroral endoscopic myotomy", "G-POEM", "Z-POEM",
        "endoscopic sleeve gastroplasty", "endoscopic suturing", "endoscopic vacuum therapy", "EVT",
        "endoluminal vacuum therapy", "EndoVAC", "endoscopic stent", "lumen-apposing metal stent", "LAMS",
        "AXIOS", "Hot AXIOS", "OTSC", "over-the-scope clip", "OverStitch", "endoscopic clipping",
        "endoscopic hemostasis", "argon plasma coagulation", "APC", "radiofrequency ablation", "RFA",
        "ERCP", "endoscopic retrograde cholangiopancreatography", "EUS-guided", "endoscopic ultrasound-guided",
        "EUS guided drainage", "EUS-guided gastroenterostomy", "EUS-guided biliary drainage", "SpyGlass",
        "cholangioscopy", "pancreatoscopy", "confocal laser endomicroscopy"
    ],
    "anastomosis_reconstruction_techniques": [
        "anastomosis", "surgical anastomosis", "anastomotic technique", "anastomotic leak", "anastomotic leakage",
        "anastomotic insufficiency", "hand-sewn anastomosis", "stapled anastomosis", "double-stapling technique",
        "single-stapling technique", "triple-stapling", "linear stapled anastomosis", "circular stapled anastomosis",
        "side-to-side anastomosis", "end-to-end anastomosis", "end-to-side anastomosis", "functional end-to-end anastomosis",
        "delta-shaped anastomosis", "overlap anastomosis", "triangulating stapling", "intracorporeal anastomosis",
        "extracorporeal anastomosis", "compression anastomosis", "magnetic compression anastomosis",
        "magnamosis", "magnetic anastomosis", "biofragmentable anastomosis ring", "BAR anastomosis",
        "Valtrac", "purse-string suture", "anvil", "transoral anvil", "OrVil", "circular stapler anastomosis",
        "Roux-en-Y anastomosis", "hepaticojejunostomy", "pancreaticojejunostomy", "gastrojejunostomy",
        "esophagojejunostomy", "colorectal anastomosis", "coloanal anastomosis", "ileocolic anastomosis",
        "Kono-S anastomosis", "barbed suture anastomosis", "reinforced anastomosis", "buttressed staple line"
    ],
    "disease_problem_enrichment": [
        "anastomotic leak", "anastomotic leakage", "anastomotic insufficiency", "pancreatic fistula", "postoperative pancreatic fistula",
        "bile leak", "biliary leak", "bile duct injury", "surgical site infection", "organ space infection", "intra-abdominal abscess",
        "postoperative hemorrhage", "postoperative bleeding", "peritonitis", "ileus", "bowel obstruction", "ischemia",
        "perfusion", "tumor margin", "resection margin", "lymph node", "sentinel lymph node", "colorectal cancer",
        "rectal cancer", "colon cancer", "gastric cancer", "esophageal cancer", "pancreatic cancer", "cholangiocarcinoma",
        "hepatocellular carcinoma", "liver metastases", "peritoneal metastases", "peritoneal carcinomatosis",
        "Crohn", "ulcerative colitis", "inflammatory bowel disease", "diverticulitis", "appendicitis",
        "cholecystitis", "choledocholithiasis", "pancreatitis", "hernia", "obesity", "GERD", "achalasia"
    ],
}

# -----------------------------------------------------------------------------
# Technology / physics / device / software modules
# -----------------------------------------------------------------------------
TECH_MODULES: Dict[str, List[str]] = {
    # Broad robotic terms are intentionally retained because this first module produced a plausible
    # count in testing and robotics is a core target domain. Device/platform names are included to
    # capture older and newer systems.
    "robotic_systems_platforms": [
        "Robotics[Mesh]", "Robotic Surgical Procedures[Mesh]",
        "robotic surgery", "robot-assisted surgery", "robotic-assisted surgery", "robotically assisted surgery",
        "computer-assisted surgery", "master-slave surgical robot", "surgical telemanipulator", "telesurgery",
        "teleoperated surgery", "teleoperation surgery", "remote surgery", "haptic feedback robotic surgery",
        "da Vinci", "da Vinci Si", "da Vinci Xi", "da Vinci X", "da Vinci SP", "Intuitive Surgical",
        "EndoWrist", "Firefly fluorescence", "SureForm", "EndoWrist stapler",
        "Hugo RAS", "Hugo robotic", "Medtronic Hugo", "Versius", "CMR Versius", "Senhance", "ALF-X",
        "ZEUS robotic", "AESOP robotic", "Mona robotic", "MiroSurge", "Raven surgical robot",
        "Revo-i", "Revo I", "hinotori", "Hinotori", "Avatera", "Dexter surgical robot", "Distalmotion Dexter",
        "Toumai", "TuoMai", "KangDuo", "Kangduo", "MicroHand S", "Shurui", "Edge MP1000", "Bitrack",
        "Robodoc", "Flex robotic system", "single-port robotic", "single port robotic", "multiport robotic",
        "robotic stapler", "robotic console", "robotic arm surgery", "telepresence surgery"
    ],

    # No generic laparoscopic/endoscopic terms here; only access platforms, ports, insufflation/smoke systems, and named devices.
    "laparoscopic_access_ports_insufflation": [
        # No generic Laparoscopy[Mesh] here: otherwise the general surgery block AND this block becomes almost all laparoscopic literature.
        "single incision laparoscopic surgery", "SILS", "SILS port", "laparoendoscopic single-site surgery", "LESS",
        "single-port laparoscopy", "single port laparoscopy", "single-site laparoscopy", "reduced-port laparoscopy",
        "natural orifice transluminal endoscopic surgery", "NOTES", "transumbilical laparoscopy",
        "optical trocar", "bladeless trocar", "balloon trocar", "balloon fixation trocar", "fixation balloon cannula",
        "balloon cannula", "Hasson trocar", "Veress needle", "radially expanding trocar", "radially expanding access",
        "AirSeal", "SurgiQuest", "valveless trocar", "valveless insufflation", "smoke evacuation system",
        "pneumoperitoneum insufflator", "laparoscopic insufflator", "high-flow insufflation",
        "GelPOINT", "GelPOINT Mini", "GelPort", "Alexis wound retractor", "Alexis laparoscopic system",
        "Lap-Protector", "hand-assist device", "hand port", "GelSeal", "TriPort", "QuadPort",
        "VersaOne", "VersaOne optical trocar", "VersaStep", "Visiport", "Optiview", "Endopath XCEL",
        "Kii access", "Kii balloon", "Applied Medical Kii", "Endo Catch", "specimen retrieval bag", "wound protector"
    ],

    # Tightened: removed broad single words such as electric/electrical/electromagnetic/coagulation/ultrasound/thermal/laser.
    # Kept named energy devices and precise technology phrases.
    "energy_electrosurgery_ultrasonic_thermal": [
        "electrosurgical generator", "electrosurgical unit", "monopolar electrosurgery", "bipolar electrosurgery",
        "advanced bipolar", "advanced bipolar vessel sealing", "bipolar vessel sealing", "vessel sealing system",
        "vessel sealing device", "electrothermal bipolar vessel sealing", "thermal spread", "thermal injury electrosurgery",
        "surgical energy device", "energy-based surgical device", "electrosurgical dissection", "electrosurgical coagulation",
        "ultrasonic scalpel", "ultrasonic shears", "ultrasonic surgical aspirator", "ultrasonic dissection",
        "ultrasonic dissector", "ultrasonic energy device", "ultrasonic coagulating shears", "ultrasonic vibration device",
        "HARMONIC scalpel", "Harmonic Scalpel", "Harmonic ACE", "Harmonic Focus", "Harmonic HD", "Ethicon Harmonic",
        "LigaSure", "LigaSure Maryland", "LigaSure Atlas", "LigaSure V", "Valleylab", "ForceTriad", "Valleylab ForceTriad",
        "EnSeal", "ENSEAL", "Thunderbeat", "THUNDERBEAT", "SonoSurg", "Sonicision", "Voyant", "Biclamp", "BiClamp",
        "ERBE VIO", "VIO 3", "VIO300", "VIO 300", "ICC 350", "argon plasma coagulation", "APC probe",
        "argon beam coagulator", "argon beam coagulation", "Aquamantys", "PlasmaJet", "J-Plasma", "helium plasma scalpel",
        "radiofrequency ablation", "RFA ablation", "microwave ablation", "MWA ablation", "irreversible electroporation",
        "NanoKnife", "cryoablation", "cryosurgery", "laser ablation", "Nd:YAG laser", "CO2 laser", "diode laser",
        "waterjet dissection", "water-jet dissection", "HydroJet", "CUSA", "Cavitron ultrasonic surgical aspirator",
        "TissueLink", "Habib sealer", "saline-linked radiofrequency", "thermal ablation", "electroporation ablation"
    ],

    "staplers_anastomotic_devices_sutures": [
        "Surgical Stapling[Mesh]",
        "surgical stapler", "stapling device", "powered stapler", "powered stapling", "linear stapler", "circular stapler",
        "endoscopic stapler", "articulating stapler", "curved cutter stapler", "linear cutter", "staple line reinforcement",
        "staple-line reinforcement", "staple line buttress", "buttressed staple line", "reinforced staple line",
        "purse-string device", "purse-string suture", "transoral anvil",
        "EEA stapler", "EEA circular stapler", "DST Series EEA", "DST EEA", "OrVil", "CEEA", "CDH stapler",
        "GIA stapler", "Endo GIA", "EndoGIA", "TA stapler", "TLH stapler", "Contour stapler",
        "Echelon stapler", "ECHELON stapler", "Echelon Flex", "ECHELON FLEX", "ECHELON GST", "Echelon circular",
        "Signia stapling", "Signia stapler", "Tri-Staple", "iDrive Ultra", "powered vascular stapler",
        "compression anastomosis", "compression anastomosis device", "magnetic compression anastomosis",
        "magnetic anastomosis", "magnetic anastomosis system", "magnamosis", "MAGNAMOSIS", "NiTi anastomosis",
        "nitinol anastomosis", "compression ring", "biofragmentable anastomosis ring", "BAR anastomosis", "Valtrac",
        "anastomotic ring", "sutureless anastomosis", "anastomotic coupler", "anastomotic device", "stent anastomosis",
        "barbed suture", "barbed suturing", "V-Loc", "Stratafix", "Quill suture", "self-locking suture", "knotless suture"
    ],

    "imaging_fluorescence_navigation_tracking": [
        "Image Processing, Computer-Assisted[Mesh]", "Indocyanine Green[Mesh]",
        "fluorescence imaging", "fluorescence-guided surgery", "fluorescence angiography", "near-infrared fluorescence",
        "near infrared fluorescence", "NIR fluorescence", "NIRF", "indocyanine green", "ICG fluorescence", "ICG angiography",
        "Firefly fluorescence", "PINPOINT fluorescence", "SPY imaging", "SPY-PHI", "Novadaq", "LUNA fluorescence",
        "fluorescein fluorescence", "methylene blue fluorescence", "5-ALA fluorescence", "autofluorescence imaging",
        "hyperspectral imaging", "multispectral imaging", "spectral imaging", "Raman spectroscopy", "Raman imaging",
        "optical coherence tomography", "OCT imaging", "confocal laser endomicroscopy", "CLE imaging",
        "photoacoustic imaging", "thermal imaging", "thermography imaging", "perfusion imaging", "laser speckle imaging",
        "laser Doppler", "intraoperative Doppler", "laparoscopic ultrasound", "intraoperative ultrasound",
        "contrast-enhanced ultrasound", "CEUS", "elastography", "ultrasound elastography",
        "image-guided surgery", "surgical navigation", "computer-assisted navigation", "electromagnetic tracking",
        "optical tracking", "fiducial registration", "deformable registration", "augmented reality surgery",
        "mixed reality surgery", "virtual reality surgical planning", "Microsoft HoloLens", "holographic navigation",
        "3D reconstruction", "three-dimensional reconstruction", "3D surgical model", "virtual surgical planning",
        "fluorescence cholangiography", "near-infrared cholangiography", "radioguided surgery", "radio-guided surgery",
        "gamma probe", "sentinel lymph node mapping", "radiotracer mapping", "SPECT navigation", "PET-guided surgery",
        "Cerenkov imaging", "beta probe", "CT-guided ablation", "MRI-guided ablation", "ultrasound-guided ablation"
    ],

    "ai_software_computer_vision_digital": [
        "Artificial Intelligence[Mesh]", "Machine Learning[Mesh]", "Deep Learning[Mesh]", "Software[Mesh]",
        "artificial intelligence", "machine learning", "deep learning", "neural network", "convolutional neural network", "CNN model",
        "transformer model", "foundation model", "large language model", "LLM", "natural language processing", "NLP",
        "computer vision", "machine vision", "surgical vision", "image segmentation", "semantic segmentation",
        "instance segmentation", "object detection", "instrument detection", "instrument tracking", "surgical phase recognition",
        "workflow recognition", "surgical workflow recognition", "action recognition", "gesture recognition", "skill assessment",
        "surgical video analysis", "operative video analysis", "laparoscopic video analysis", "endoscopic video analysis",
        "pose estimation", "SLAM", "simultaneous localization and mapping", "decision support system",
        "clinical decision support", "prediction model", "predictive model", "risk model", "nomogram model",
        "digital surgery", "digital operating room", "operating room analytics", "surgical data science", "digital twin",
        "surgical simulation", "virtual patient", "finite element model", "finite element analysis", "computational fluid dynamics",
        "CFD model", "computer-aided design", "CAD model", "3D printing", "three-dimensional printing", "additive manufacturing",
        "rapid prototyping", "patient-specific model", "telementoring", "teleoperation", "remote proctoring"
    ],

    # Tightened: removed generic mesh/polymer/gel terms; retained device/material/product phrases and named products.
    "materials_mesh_sealants_hemostats_biomaterials": [
        "Biocompatible Materials[Mesh]", "Hydrogels[Mesh]", "Tissue Adhesives[Mesh]", "Hemostatics[Mesh]", "Surgical Mesh[Mesh]",
        "surgical biomaterial", "biomaterial implant", "hydrogel sealant", "smart hydrogel", "adhesive hydrogel", "PEG hydrogel",
        "polyethylene glycol hydrogel", "surgical adhesive", "tissue adhesive", "surgical glue", "fibrin glue", "fibrin sealant",
        "Tisseel", "Evicel", "Vistaseal", "Beriplast", "Coseal", "CoSeal", "BioGlue", "Progel", "DuraSeal",
        "TachoSil", "TachoComb", "Surgicel", "Floseal", "FloSeal", "Surgiflo", "Hemopatch", "Veriset",
        "EVARREST", "Arista", "Hemospray", "PuraStat", "EndoClot", "cyanoacrylate glue", "N-butyl cyanoacrylate",
        "collagen patch", "hemostatic patch", "hemostatic powder", "hemostatic matrix", "fibrin patch",
        "surgical mesh", "polypropylene mesh", "ePTFE mesh", "expanded polytetrafluoroethylene mesh", "PTFE mesh",
        "PVDF mesh", "polyester mesh", "composite mesh", "biologic mesh", "biosynthetic mesh", "absorbable mesh",
        "resorbable mesh", "self-gripping mesh", "coated mesh", "anti-adhesion barrier", "adhesion barrier",
        "Seprafilm", "Interceed", "Parietex", "Proceed mesh", "Physiomesh", "Ventralex", "ProGrip", "DynaMesh",
        "GORE-TEX mesh", "DualMesh", "Strattice", "Surgisis", "Permacol", "Phasix", "P4HB mesh", "OviTex", "Vicryl mesh",
        "tissue engineering scaffold", "nanomaterial coating", "antimicrobial coating", "drug-eluting coating"
    ],

    # Tightened: removed broad pressure/sensor/magnetic alone; kept concrete sensing/measurement/navigation phrases.
    "sensors_physics_magnetic_electromagnetic_pressure": [
        "Biosensing Techniques[Mesh]", "Electromagnetic Fields[Mesh]",
        "magnetic compression anastomosis", "magnetic anastomosis", "magnetic sphincter augmentation", "magnetic surgical instrument",
        "magnetic navigation", "magnetic actuation", "magnetically actuated", "electromagnetic tracking", "electromagnetic navigation",
        "electrical impedance", "bioimpedance", "impedance spectroscopy", "dielectric spectroscopy",
        "pressure sensor", "force sensor", "strain sensor", "tactile sensor", "haptic feedback", "force feedback", "tactile feedback",
        "intraluminal pressure", "anastomotic pressure", "pressure monitoring", "tensiometry", "anastomotic tension",
        "flow sensor", "perfusion sensor", "oxygen sensor", "pH sensor", "temperature sensor", "thermal sensor",
        "accelerometer", "gyroscope", "inertial measurement unit", "RFID tracking", "wireless sensor", "Bluetooth sensor",
        "smart instrument", "microfluidic sensor", "lab-on-chip", "MEMS sensor", "spectrometer", "spectroscopy probe",
        "optical fiber sensor", "fiberoptic sensor", "fiber-optic sensor", "laser Doppler flowmetry", "acoustic sensor",
        "ultrasonic sensor", "vibration sensor", "elastography", "tissue stiffness measurement"
    ],

    # Specific endoluminal/endoscopic procedures and devices. No generic endoscopic/endoscopy/endoscope term.
    "specific_endoscopic_endoluminal_procedures_devices": [
        "endoscopic balloon dilation", "endoscopic balloon dilatation", "balloon dilation", "balloon dilatation",
        "pneumatic dilation", "pneumatic dilatation", "achalasia balloon dilation", "papillary balloon dilation",
        "endoscopic papillary balloon dilation", "CRE balloon", "through-the-scope balloon", "TTS balloon",
        "esophageal dilation", "esophageal dilatation", "pyloric balloon dilation", "colonic balloon dilation",
        "endoscopic manometry", "high-resolution manometry", "HRM", "esophageal manometry", "anorectal manometry",
        "pH monitoring", "pH metry", "pH-metry", "impedance pH monitoring", "Bravo pH", "wireless pH monitoring",
        "EndoFLIP", "functional lumen imaging probe", "FLIP panometry", "impedance planimetry",
        "endoscopic submucosal dissection", "ESD", "endoscopic mucosal resection", "EMR", "piecemeal EMR",
        "endoscopic full-thickness resection", "EFTR", "full-thickness resection device", "FTRD",
        "submucosal tunneling endoscopic resection", "STER", "peroral endoscopic myotomy", "POEM",
        "gastric peroral endoscopic myotomy", "G-POEM", "Z-POEM", "endoscopic sleeve gastroplasty",
        "endoscopic suturing", "OverStitch", "endoscopic vacuum therapy", "endoluminal vacuum therapy", "EndoVAC",
        "endoscopic stent", "self-expandable metal stent", "SEMS", "covered metal stent", "biliary stent",
        "pancreatic stent", "colonic stent", "esophageal stent", "duodenal stent", "lumen-apposing metal stent", "LAMS",
        "AXIOS stent", "Hot AXIOS", "Niti-S stent", "over-the-scope clip", "OTSC", "Padlock clip",
        "through-the-scope clip", "TTS clip", "endoscopic clipping", "endoscopic hemostasis",
        "endoscopic band ligation", "endoscopic snare resection", "endoscopic knife", "DualKnife", "IT knife", "HookKnife",
        "HybridKnife", "Coagrasper", "hemostatic forceps", "EUS-guided drainage", "EUS guided drainage",
        "endoscopic ultrasound-guided drainage", "EUS-guided gastroenterostomy", "EUS-guided biliary drainage",
        "endoscopic retrograde cholangiopancreatography", "ERCP", "SpyGlass cholangioscopy", "peroral cholangioscopy",
        "cholangioscopy", "pancreatoscopy", "confocal laser endomicroscopy", "probe-based confocal laser endomicroscopy"
    ],

    "perfusion_ablation_interventional_oncology": [
        "tumor ablation", "radiofrequency ablation", "RFA ablation", "microwave ablation", "MWA ablation",
        "cryoablation", "irreversible electroporation", "IRE ablation", "NanoKnife", "laser ablation", "thermal ablation",
        "high-intensity focused ultrasound", "HIFU", "focused ultrasound ablation", "image-guided ablation",
        "needle tracking ablation", "navigation ablation", "chemoembolization", "TACE", "radioembolization", "SIRT",
        "Y-90 radioembolization", "yttrium radioembolization", "brachytherapy", "intraoperative radiotherapy", "IORT",
        "HIPEC", "hyperthermic intraperitoneal chemotherapy", "PIPAC", "pressurized intraperitoneal aerosol chemotherapy",
        "electrochemotherapy", "photodynamic therapy", "PDT", "isolated hepatic perfusion", "portal vein embolization",
        "hepatic venous deprivation", "radiological intervention"
    ],
}


# -----------------------------------------------------------------------------
# Query safety: avoid circular/broad duplicates
# -----------------------------------------------------------------------------
# These terms are allowed only inside specific phrases/device names. They should not appear as
# standalone technical triggers, because they explode PubMed counts and create circular queries.
FORBIDDEN_STANDALONE_TECH_TERMS = {
    "endoscopic", "endoscopy", "endoscope", "laparoscopic", "laparoscopy", "robotic",
    "ultrasound", "ultrasonic", "electric", "electrical", "electromagnetic", "magnetic",
    "thermal", "coagulation", "laser", "camera", "vision", "image", "imaging",
    "sensor", "pressure", "mesh", "gel", "stent", "clip", "software", "algorithm",
}

# Some surgery modules are themselves technology/procedure modules. Do not pair them with the
# matching technology module, because that produces A AND A queries rather than surgery-context AND technology.
SKIP_MODULE_PAIRS = {
    ("advanced_endoscopy_endoluminal", "specific_endoscopic_endoluminal_procedures_devices"),
    ("anastomosis_reconstruction_techniques", "staplers_anastomotic_devices_sutures"),
}

# These exact broad terms should not be duplicated between surgery and technology blocks.
BROAD_DUPLICATE_TERMS = {
    "SILS", "LESS", "NOTES", "laparoscopy", "laparoscopic", "robotic surgery", "trocar",
    "pneumoperitoneum", "insufflation", "endoscopic stent", "OTSC", "OverStitch", "LAMS", "AXIOS",
    "argon plasma coagulation", "APC", "radiofrequency ablation", "RFA", "ERCP", "SpyGlass",
}


def validate_query_modules() -> None:
    """Print safety warnings before starting long PubMed extraction."""
    # 1) Standalone overly broad tech triggers.
    offenders = []
    for mod, terms in TECH_MODULES.items():
        for term in terms:
            raw = term.replace('[Mesh]', '').replace('[MeSH Terms]', '').strip().strip('"')
            if raw.lower() in FORBIDDEN_STANDALONE_TECH_TERMS:
                offenders.append((mod, term))
    if offenders:
        print("WARNING: standalone broad technology terms detected:")
        for mod, term in offenders:
            print(f"  - {mod}: {term}")

    # 2) Exact overlaps between surgery and tech modules. These are not always fatal, but broad ones are removed in build_queries().
    for s_name, s_terms in SURGERY_MODULES.items():
        s_set = {x.lower().strip() for x in s_terms}
        for t_name, t_terms in TECH_MODULES.items():
            overlap = sorted(s_set.intersection({x.lower().strip() for x in t_terms}))
            broad_overlap = [x for x in overlap if x in {b.lower() for b in BROAD_DUPLICATE_TERMS}]
            if broad_overlap:
                print(f"NOTE: broad overlap will be filtered for {s_name} AND {t_name}: {broad_overlap[:10]}")


def filter_tech_terms_for_pair(s_terms: List[str], t_terms: List[str]) -> List[str]:
    """Remove exact broad duplicates from the tech side for a given surgery-tech pair."""
    s_lower = {x.lower().strip() for x in s_terms}
    broad_lower = {x.lower() for x in BROAD_DUPLICATE_TERMS}
    out = []
    for term in t_terms:
        key = term.lower().strip()
        if key in s_lower and key in broad_lower:
            continue
        out.append(term)
    return out

# Optional disease enrichment with surgery terms, to catch disease-first records.
SURGICAL_QUALIFIER = [
    "surgery", "surgical", "operative", "operation", "procedure", "resection", "repair", "anastomosis",
    "laparoscopic surgery", "robotic surgery", "endoscopic resection", "minimally invasive surgery"
]

# -----------------------------------------------------------------------------
# E-utilities helpers
# -----------------------------------------------------------------------------

def eutils_post(endpoint: str, payload: Dict[str, str], timeout: int = 90) -> requests.Response:
    payload = dict(payload)
    payload.update({"tool": TOOL, "email": EMAIL})
    if API_KEY:
        payload["api_key"] = API_KEY
    delay = 0.11 if API_KEY else 0.35
    for attempt in range(1, 6):
        try:
            r = requests.post(f"{EUTILS}/{endpoint}", data=payload, timeout=timeout)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            time.sleep(delay)
            return r
        except Exception as exc:
            if attempt == 5:
                raise
            wait = min(2 ** attempt, 30)
            print(f"Retry {attempt}/5 after error: {exc}. Waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def parse_esearch_xml(content: bytes) -> Tuple[int, List[str]]:
    """Parse PubMed ESearch XML. XML is more robust than JSON for very long queries."""
    root = ET.fromstring(content)
    count_text = get_text(root.find("Count"))
    count = int(count_text) if count_text.isdigit() else 0
    ids = [get_text(x) for x in root.findall("IdList/Id") if get_text(x)]
    return count, ids


def esearch_count(query: str) -> int:
    """Return PubMed hit count for one query."""
    first = eutils_post("esearch.fcgi", {
        "db": DB, "term": query, "retmode": "xml", "retmax": "0", "usehistory": "n"
    })
    count, _ = parse_esearch_xml(first.content)
    return count


def esearch_fetch_ids_no_split(query: str, expected_count: Optional[int] = None) -> List[str]:
    """Fetch IDs for a query that is expected to be below the PubMed 9999-ID practical cap."""
    if expected_count is None:
        expected_count = esearch_count(query)

    # Defensive cap: this function should only be used after splitting. If it is still too large,
    # fetch what PubMed exposes and warn loudly rather than silently pretending it is complete.
    if expected_count > DATE_SPLIT_THRESHOLD:
        print(f"WARNING: unsplit query still has count={expected_count}; fetching first {SEARCH_RETMAX} only. Consider smaller splits.")

    pmids: List[str] = []
    max_to_fetch = min(expected_count, SEARCH_RETMAX)
    if max_to_fetch == 0:
        return pmids

    for retstart in range(0, max_to_fetch, SEARCH_RETMAX):
        res = eutils_post("esearch.fcgi", {
            "db": DB, "term": query, "retmode": "xml",
            "retstart": str(retstart), "retmax": str(SEARCH_RETMAX), "usehistory": "n"
        })
        _, ids = parse_esearch_xml(res.content)
        pmids.extend(ids)
    return pmids


def esearch_pmids(query: str) -> Tuple[int, List[str]]:
    """Return PubMed count and PMIDs for one subquery with automatic date splitting.

    Logic:
    1. Count full 1980-2025 query.
    2. If count <= 9999, fetch directly.
    3. If count > 9999, run the same query in 10 publication-date windows.
    4. If a 10-window segment is still >9999, split by individual years.
    5. If a year is still >9999, split by months.

    This avoids the practical PubMed/ESearch 9999-ID cap that produced unique_ids=9999.
    """
    total_count = esearch_count(query)
    if total_count <= DATE_SPLIT_THRESHOLD:
        return total_count, esearch_fetch_ids_no_split(query, total_count)

    print(f"  Large query count={total_count}; splitting into 10 date windows...")
    all_pmids: List[str] = []

    for win_start, win_end in DATE_WINDOWS_10:
        win_query = clean_query(f"({query}) AND {pub_date_filter(win_start, win_end)}")
        win_count = esearch_count(win_query)
        print(f"    window {win_start}-{win_end}: count={win_count}")
        if win_count == 0:
            continue
        if win_count <= DATE_SPLIT_THRESHOLD:
            all_pmids.extend(esearch_fetch_ids_no_split(win_query, win_count))
            continue

        # Split oversized window by year.
        start_year = int(win_start[:4])
        end_year = int(win_end[:4])
        print(f"      window still >{DATE_SPLIT_THRESHOLD}; splitting by year...")
        for y_start, y_end in year_windows(start_year, end_year):
            y_query = clean_query(f"({query}) AND {pub_date_filter(y_start, y_end)}")
            y_count = esearch_count(y_query)
            print(f"        year {y_start[:4]}: count={y_count}")
            if y_count == 0:
                continue
            if y_count <= DATE_SPLIT_THRESHOLD:
                all_pmids.extend(esearch_fetch_ids_no_split(y_query, y_count))
                continue

            # Last-resort split by month.
            year = int(y_start[:4])
            print(f"          year still >{DATE_SPLIT_THRESHOLD}; splitting by month...")
            for m_start, m_end in month_windows(year):
                m_query = clean_query(f"({query}) AND {pub_date_filter(m_start, m_end)}")
                m_count = esearch_count(m_query)
                print(f"            month {m_start[:7]}: count={m_count}")
                if m_count == 0:
                    continue
                all_pmids.extend(esearch_fetch_ids_no_split(m_query, m_count))

    return total_count, all_pmids


def get_text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def parse_pub_date(article: ET.Element) -> Tuple[str, str]:
    journal_issue = article.find(".//JournalIssue/PubDate")
    year = get_text(journal_issue.find("Year")) if journal_issue is not None else ""
    medline_date = get_text(journal_issue.find("MedlineDate")) if journal_issue is not None else ""
    month = get_text(journal_issue.find("Month")) if journal_issue is not None else ""
    day = get_text(journal_issue.find("Day")) if journal_issue is not None else ""
    if not year and medline_date:
        m = re.search(r"(19|20)\d{2}", medline_date)
        year = m.group(0) if m else ""
    pubdate = "-".join(x for x in [year, month, day] if x)
    return year, pubdate


def parse_article(pubmed_article: ET.Element, query_names: List[str]) -> Dict[str, str]:
    medline = pubmed_article.find("MedlineCitation")
    article = medline.find("Article") if medline is not None else None
    if medline is None or article is None:
        return {}

    pmid = get_text(medline.find("PMID"))
    year, pubdate = parse_pub_date(article)
    journal = get_text(article.find("Journal/Title"))
    journal_iso = get_text(article.find("Journal/ISOAbbreviation"))
    title = html.unescape(get_text(article.find("ArticleTitle")))

    abstract_parts = []
    for abst in article.findall("Abstract/AbstractText"):
        label = abst.attrib.get("Label", "")
        txt = html.unescape(get_text(abst))
        if txt:
            abstract_parts.append(f"{label}: {txt}" if label else txt)
    abstract = "\n".join(abstract_parts)

    # DOI and article IDs
    doi = ""
    pmc = ""
    for aid in pubmed_article.findall("PubmedData/ArticleIdList/ArticleId"):
        typ = aid.attrib.get("IdType", "")
        val = get_text(aid)
        if typ == "doi" and not doi:
            doi = val
        elif typ == "pmc" and not pmc:
            pmc = val

    pub_types = "; ".join(get_text(pt) for pt in article.findall("PublicationTypeList/PublicationType"))
    mesh_terms = "; ".join(get_text(mh.find("DescriptorName")) for mh in medline.findall("MeshHeadingList/MeshHeading"))
    keywords = "; ".join(get_text(k) for k in medline.findall("KeywordList/Keyword"))

    authors = []
    affiliations = []
    for author in article.findall("AuthorList/Author"):
        last = get_text(author.find("LastName"))
        fore = get_text(author.find("ForeName"))
        coll = get_text(author.find("CollectiveName"))
        name = coll if coll else " ".join(x for x in [fore, last] if x)
        if name:
            authors.append(name)
        for aff in author.findall("AffiliationInfo/Affiliation"):
            a = get_text(aff)
            if a:
                affiliations.append(a)

    return {
        "pmid": pmid,
        "year": year,
        "publication_date": pubdate,
        "journal": journal,
        "journal_iso": journal_iso,
        "title": title,
        "abstract": abstract,
        "doi": doi,
        "pmc": pmc,
        "publication_types": pub_types,
        "mesh_terms": mesh_terms,
        "keywords": keywords,
        "authors": "; ".join(authors),
        "first_author": authors[0] if authors else "",
        "last_author": authors[-1] if authors else "",
        "affiliations": " | ".join(dict.fromkeys(affiliations)),
        "matched_query_modules": "; ".join(sorted(set(query_names))),
        "n_matched_query_modules": str(len(set(query_names))),
    }


def efetch_articles(pmids: List[str], pmid_to_queries: Dict[str, List[str]]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for i in tqdm(range(0, len(pmids), FETCH_BATCH_SIZE), desc="Fetching PubMed XML"):
        batch = pmids[i:i + FETCH_BATCH_SIZE]
        r = eutils_post("efetch.fcgi", {
            "db": DB, "id": ",".join(batch), "retmode": "xml"
        })
        root = ET.fromstring(r.content)
        for art in root.findall("PubmedArticle"):
            pmid = get_text(art.find("MedlineCitation/PMID"))
            rec = parse_article(art, pmid_to_queries.get(pmid, []))
            if rec:
                records.append(rec)
    return records


def build_queries() -> Dict[str, str]:
    queries: Dict[str, str] = {}
    tech_names = list(TECH_MODULES.keys())

    # Main module combinations: each requires surgery module AND tech module.
    for s_name, s_terms in SURGERY_MODULES.items():
        s_block = or_block(s_terms)
        for t_name in tech_names:
            if (s_name, t_name) in SKIP_MODULE_PAIRS:
                continue
            filtered_t_terms = filter_tech_terms_for_pair(s_terms, TECH_MODULES[t_name])
            if not filtered_t_terms:
                continue
            t_block = or_block(filtered_t_terms)
            name = f"{s_name}__AND__{t_name}"
            queries[name] = clean_query(f"({s_block}) AND ({t_block}) AND {DATE_ABSTRACT_HUMAN}")

    # Disease enrichment: disease/problem AND surgical qualifier AND technology.
    disease_block = or_block(SURGERY_MODULES["disease_problem_enrichment"])
    qualifier_block = or_block(SURGICAL_QUALIFIER)
    for t_name in tech_names:
        t_block = or_block(TECH_MODULES[t_name])
        name = f"disease_problem_plus_surgical_qualifier__AND__{t_name}"
        queries[name] = clean_query(f"({disease_block}) AND ({qualifier_block}) AND ({t_block}) AND {DATE_ABSTRACT_HUMAN}")

    return queries


def main() -> None:
    print("Building expanded modular PubMed queries...")
    validate_query_modules()
    queries = build_queries()
    print(f"Number of subqueries: {len(queries)}")

    with open(OUTPUT_QUERY_TXT, "w", encoding="utf-8") as f:
        f.write(f"PubMed visceral surgery x technology expanded extraction\n")
        f.write(f"Date range: {START_DATE} to {END_DATE}\n")
        f.write(f"Created: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Subqueries: {len(queries)}\n\n")
        for name, q in queries.items():
            f.write(f"### {name}\n{q}\n\n")

    with open(OUTPUT_QUERY_JSON, "w", encoding="utf-8") as f:
        json.dump({"queries": queries, "surgery_modules": SURGERY_MODULES, "tech_modules": TECH_MODULES}, f, ensure_ascii=False, indent=2)

    pmid_to_queries: Dict[str, List[str]] = defaultdict(list)
    query_stats = []

    for name, q in tqdm(queries.items(), desc="Searching PubMed subqueries"):
        try:
            count, pmids = esearch_pmids(q)
            unique_pmids = sorted(set(pmids))
            for pmid in unique_pmids:
                pmid_to_queries[pmid].append(name)
            query_stats.append({"query_name": name, "pubmed_count": count, "retrieved_unique_pmids": len(unique_pmids)})
            print(f"{name}: count={count}, unique_ids={len(unique_pmids)}")
        except Exception as exc:
            query_stats.append({"query_name": name, "pubmed_count": "ERROR", "retrieved_unique_pmids": 0, "error": str(exc)})
            print(f"ERROR in {name}: {exc}")

    all_pmids = sorted(pmid_to_queries.keys(), key=lambda x: int(x))
    print(f"Total unique PMIDs across all modules: {len(all_pmids)}")

    records = efetch_articles(all_pmids, pmid_to_queries) if all_pmids else []
    df = pd.DataFrame(records)
    if not df.empty:
        # Defensive sorting and year conversion.
        df["year_numeric"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.sort_values(["year_numeric", "journal", "title", "pmid"], na_position="last").drop(columns=["year_numeric"])

    stats_df = pd.DataFrame(query_stats)
    meta_df = pd.DataFrame([
        {"field": "created", "value": datetime.now().isoformat(timespec="seconds")},
        {"field": "date_range", "value": f"{START_DATE} to {END_DATE}"},
        {"field": "n_subqueries", "value": len(queries)},
        {"field": "n_unique_pmids", "value": len(all_pmids)},
        {"field": "output_prefix", "value": OUTPUT_PREFIX},
        {"field": "logic", "value": "(visceral surgery/procedure/disease module) AND (technical/physics/device/software/material module) AND abstract/date/human filter"},
    ])

    print(f"Writing {OUTPUT_XLSX} and {OUTPUT_CSV}...")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="pubmed_records")
        stats_df.to_excel(writer, index=False, sheet_name="query_stats")
        meta_df.to_excel(writer, index=False, sheet_name="metadata")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Excel: {OUTPUT_XLSX}")
    print(f"CSV:   {OUTPUT_CSV}")
    print(f"Query text: {OUTPUT_QUERY_TXT}")
    print(f"Query JSON: {OUTPUT_QUERY_JSON}")


if __name__ == "__main__":
    main()
