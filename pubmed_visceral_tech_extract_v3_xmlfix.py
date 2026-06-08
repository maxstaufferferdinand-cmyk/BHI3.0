#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PubMed extraction v2: visceral/abdominal surgery operations/diseases/techniques
AND concrete surgical technology / physics / device / software / material terms, 1980-2025.

Why v2?
- Uses modular subqueries instead of one oversized query, to reduce missed records and PubMed query-length/10k-limit problems.
- Every subquery requires at least one visceral surgery/procedure/disease term AND at least one technical/device/physics term.
- Exports a deduplicated Excel + CSV with metadata and which query modules matched each PMID.

Install:
    python -m pip install requests pandas openpyxl tqdm

Recommended environment variables:
    setx NCBI_EMAIL "n12028114@students.meduniwien.ac.at"
    setx NCBI_API_KEY "None"

Run:
    python pubmed_visceral_tech_extract_v2.py
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
TOOL = "visceral_surgery_tech_bridge_extractor_v3_xml_esearch"

SEARCH_RETMAX = 10000       # PubMed ESearch batch size for IDs
FETCH_BATCH_SIZE = 200      # EFetch XML batch size
START_DATE = "1980/01/01"
END_DATE = "2025/12/31"
OUTPUT_PREFIX = "pubmed_visceral_surgery_technology_expanded_1980_2025_v3"
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

SURGERY_CONTEXT_MESH = [
    "General Surgery[Mesh]", "Digestive System Surgical Procedures[Mesh]",
    "Gastrointestinal Surgical Procedures[Mesh]", "Minimally Invasive Surgical Procedures[Mesh]",
    "Laparoscopy[Mesh]", "Robotic Surgical Procedures[Mesh]", "Endoscopy, Digestive System[Mesh]",
    "Anastomosis, Surgical[Mesh]", "Surgical Stapling[Mesh]", "Hepatectomy[Mesh]",
    "Pancreatectomy[Mesh]", "Colectomy[Mesh]", "Cholecystectomy[Mesh]", "Bariatric Surgery[Mesh]",
    "Hernia, Inguinal/surgery[Mesh]", "Hernia, Ventral/surgery[Mesh]", "Esophagectomy[Mesh]",
    "Gastrectomy[Mesh]", "Appendectomy[Mesh]"
]

# -----------------------------------------------------------------------------
# Visceral / abdominal surgery modules
# -----------------------------------------------------------------------------
SURGERY_MODULES: Dict[str, List[str]] = {
    "general_minimally_invasive_access": SURGERY_CONTEXT_MESH + [
        "visceral surgery", "abdominal surgery", "digestive surgery", "gastrointestinal surgery", "GI surgery",
        "general surgery", "upper gastrointestinal surgery", "upper GI surgery", "lower gastrointestinal surgery",
        "minimally invasive surgery", "laparoscopic surgery", "laparoscopy", "laparoscopic", "robotic surgery",
        "robot-assisted surgery", "robotic-assisted surgery", "single incision laparoscopic surgery", "single-port surgery",
        "single site surgery", "reduced port surgery", "SILS", "LESS", "NOTES",
        "natural orifice transluminal endoscopic surgery", "transumbilical surgery", "laparotomy",
        "open abdomen", "damage control surgery", "enhanced recovery after surgery", "ERAS",
        "trocar", "port placement", "pneumoperitoneum", "insufflation", "surgical access", "specimen extraction",
        "intracorporeal suturing", "extracorporeal suturing", "barbed suture", "endoloop", "laparoscopic suture",
        "surgical drain", "drainage", "negative pressure wound therapy", "vacuum-assisted closure"
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
    "robotic_systems_platforms": [
        "Robotics[Mesh]", "Robotic Surgical Procedures[Mesh]", "robotic", "robotics", "robot-assisted", "robotic-assisted",
        "robotically assisted", "computer-assisted surgery", "master-slave", "telemanipulator", "telesurgery",
        "da Vinci", "da Vinci Si", "da Vinci Xi", "da Vinci X", "da Vinci SP", "Intuitive Surgical",
        "EndoWrist", "Firefly", "SureForm", "Hugo RAS", "Hugo robotic", "Medtronic Hugo",
        "Versius", "CMR Versius", "Senhance", "ALF-X", "ZEUS robotic", "AESOP", "Mona", "MiroSurge",
        "Revo-i", "Revo I", "hinotori", "Hinotori", "Avatera", "Dexter", "Distalmotion Dexter",
        "Toumai", "TuoMai", "KangDuo", "Kangduo", "MicroHand S", "Shurui", "Edge MP1000",
        "Bitrack", "Robodoc", "Flex robotic system", "single-port robotic", "multiport robotic", "robotic stapler",
        "robotic arm", "robotic console", "haptic robotic", "telepresence surgery", "remote surgery"
    ],
    "laparoscopic_access_ports_insufflation": [
        "Laparoscopy[Mesh]", "laparoscopic", "laparoscopy", "minimally invasive", "trocar", "cannula",
        "optical trocar", "bladeless trocar", "balloon trocar", "balloon fixation", "fixation balloon",
        "Hasson trocar", "Veress needle", "pneumoperitoneum", "insufflator", "insufflation", "smoke evacuation",
        "AirSeal", "SurgiQuest", "GelPOINT", "GelPort", "Alexis wound retractor", "Lap-Protector",
        "SILS port", "SILS", "LESS", "single incision", "single port", "single-site", "TriPort", "QuadPort",
        "VersaOne", "VersaStep", "Visiport", "Optiview", "Endopath XCEL", "Kii access", "Kii balloon",
        "Applied Medical Kii", "radially expanding access", "hand-assist device", "hand port", "GelSeal",
        "balloon cannula", "access platform", "multi-instrument port", "wound protector", "specimen retrieval bag", "Endo Catch"
    ],
    "energy_electrosurgery_ultrasonic_thermal": [
        "Electrocoagulation[Mesh]", "Lasers[Mesh]", "Ultrasonics[Mesh]", "electrosurgery", "electrosurgical",
        "electric", "electrical", "electromagnetic", "monopolar", "bipolar", "advanced bipolar", "coagulation",
        "electrocoagulation", "electrocautery", "diathermy", "vessel sealing", "thermal spread", "energy device",
        "HARMONIC scalpel", "Harmonic Scalpel", "Harmonic ACE", "Harmonic Focus", "Harmonic HD", "ultrasonic scalpel",
        "ultrasonic shears", "ultrasonic dissector", "ultrasound dissector", "ultrasonic vibration", "ultrasonic vibrator",
        "LigaSure", "LigaSure Maryland", "LigaSure Atlas", "LigaSure V", "Valleylab", "ForceTriad", "Triad",
        "EnSeal", "ENSEAL", "Thunderbeat", "THUNDERBEAT", "SonoSurg", "Sonicision", "Voyant",
        "Biclamp", "BiClamp", "ERBE", "VIO 3", "VIO300", "VIO 300", "ICC 350", "APC", "argon plasma coagulation",
        "argon beam", "argon beam coagulator", "Aquamantys", "PlasmaJet", "J-Plasma", "helium plasma",
        "radiofrequency", "radiofrequency ablation", "RFA", "microwave", "microwave ablation", "MWA",
        "irreversible electroporation", "IRE", "NanoKnife", "cryoablation", "cryosurgery", "laser", "Nd:YAG",
        "CO2 laser", "diode laser", "waterjet", "water-jet", "HydroJet", "CUSA", "Cavitron ultrasonic surgical aspirator",
        "TissueLink", "Habib sealer", "saline-linked radiofrequency", "thermal ablation", "electroporation"
    ],
    "staplers_anastomotic_devices_sutures": [
        "Surgical Stapling[Mesh]", "stapler", "staplers", "stapling", "powered stapler", "powered stapling",
        "linear stapler", "circular stapler", "endoscopic stapler", "articulating stapler", "reload", "staple line",
        "staple-line reinforcement", "buttress", "buttressing", "anvil", "purse-string", "transoral anvil",
        "EEA stapler", "EEA circular stapler", "DST Series EEA", "DST EEA", "OrVil", "CEEA", "CDH stapler",
        "GIA stapler", "Endo GIA", "EndoGIA", "TA stapler", "TLH stapler", "Contour stapler",
        "Echelon", "ECHELON", "Echelon Flex", "ECHELON FLEX", "ECHELON GST", "Echelon circular",
        "Signia", "Signia stapling", "Tri-Staple", "iDrive", "iDrive Ultra", "EndoWrist stapler", "SureForm",
        "powered vascular stapler", "manual stapler", "compression anastomosis", "magnetic compression anastomosis",
        "magnetic anastomosis", "magnamosis", "MAGNAMOSIS", "NiTi", "nitinol", "compression ring",
        "biofragmentable anastomosis ring", "BAR", "Valtrac", "anastomotic ring", "linear cutter",
        "barbed suture", "V-Loc", "Stratafix", "Quill suture", "self-locking suture", "knotless suture",
        "sutureless anastomosis", "anastomotic coupler", "anastomotic device", "stent anastomosis"
    ],
    "imaging_fluorescence_navigation_tracking": [
        "Imaging, Three-Dimensional[Mesh]", "Image Processing, Computer-Assisted[Mesh]", "Fluorescence[Mesh]",
        "Indocyanine Green[Mesh]", "Magnetic Resonance Imaging[Mesh]", "fluorescence", "fluorescent",
        "fluorescence imaging", "fluorescence angiography", "near-infrared", "near infrared", "NIR", "NIRF",
        "indocyanine green", "ICG", "Firefly", "PINPOINT", "SPY imaging", "SPY-PHI", "Novadaq",
        "LUNA", "fluorescein", "methylene blue", "5-ALA", "protoporphyrin", "autofluorescence",
        "hyperspectral", "hyperspectral imaging", "multispectral", "spectral imaging", "Raman", "Raman spectroscopy",
        "optical coherence tomography", "OCT", "confocal", "confocal laser endomicroscopy", "CLE",
        "photoacoustic", "thermography", "thermal imaging", "perfusion imaging", "laser speckle", "Doppler",
        "intraoperative ultrasound", "laparoscopic ultrasound", "contrast-enhanced ultrasound", "CEUS", "elastography",
        "image-guided", "image guidance", "surgical navigation", "navigation", "computer-assisted navigation",
        "electromagnetic tracking", "optical tracking", "fiducial", "registration", "deformable registration",
        "augmented reality", "mixed reality", "virtual reality", "AR", "VR", "holographic", "Microsoft HoloLens",
        "3D reconstruction", "three-dimensional reconstruction", "3D model", "surgical planning", "virtual planning",
        "cholangiography", "fluorescence cholangiography", "radioguided", "radio-guided", "gamma probe",
        "sentinel lymph node", "radiotracer", "radionuclide", "SPECT", "PET", "Cerenkov", "beta probe",
        "nuclear", "CT-guided", "MRI-guided", "ultrasound-guided"
    ],
    "ai_software_computer_vision_digital": [
        "Artificial Intelligence[Mesh]", "Machine Learning[Mesh]", "Deep Learning[Mesh]", "Software[Mesh]",
        "Computer Simulation[Mesh]", "artificial intelligence", "AI", "machine learning", "deep learning",
        "neural network", "convolutional neural network", "CNN", "transformer", "foundation model", "large language model",
        "LLM", "natural language processing", "NLP", "computer vision", "machine vision", "surgical vision",
        "image segmentation", "semantic segmentation", "instance segmentation", "object detection", "instrument detection",
        "instrument tracking", "surgical phase recognition", "workflow recognition", "action recognition", "gesture recognition",
        "skill assessment", "video analysis", "surgical video", "pose estimation", "SLAM", "simultaneous localization and mapping",
        "algorithm", "software", "informatics", "computational", "decision support", "clinical decision support",
        "prediction model", "predictive model", "risk model", "nomogram", "digital surgery", "digital operating room",
        "operating room analytics", "OR analytics", "surgical data science", "data science", "big data",
        "digital twin", "simulation", "simulator", "virtual patient", "finite element", "finite element model",
        "computational fluid dynamics", "CFD", "computer-aided design", "CAD", "computer-aided", "3D printing",
        "three-dimensional printing", "additive manufacturing", "rapid prototyping", "patient-specific model",
        "telementoring", "telemedicine", "teleoperation", "remote proctoring", "cloud", "blockchain"
    ],
    "materials_mesh_sealants_hemostats_biomaterials": [
        "Biocompatible Materials[Mesh]", "Hydrogels[Mesh]", "Tissue Adhesives[Mesh]", "Hemostatics[Mesh]",
        "Surgical Mesh[Mesh]", "biomaterial", "biomaterials", "polymer", "polymers", "hydrogel", "gel", "smart hydrogel",
        "adhesive", "surgical adhesive", "tissue adhesive", "glue", "sealant", "fibrin glue", "fibrin sealant",
        "Tisseel", "Evicel", "Vistaseal", "Beriplast", "Coseal", "CoSeal", "BioGlue", "Progel", "DuraSeal",
        "TachoSil", "TachoComb", "Surgicel", "Floseal", "FloSeal", "Surgiflo", "Hemopatch", "Veriset",
        "EVARREST", "Arista", "Hemospray", "PuraStat", "EndoClot", "cyanoacrylate", "N-butyl cyanoacrylate",
        "PEG hydrogel", "polyethylene glycol", "collagen patch", "hemostatic patch", "hemostatic powder",
        "mesh", "surgical mesh", "polypropylene mesh", "ePTFE", "expanded polytetrafluoroethylene", "PTFE",
        "PVDF", "polyester mesh", "composite mesh", "biologic mesh", "biosynthetic mesh", "absorbable mesh",
        "resorbable mesh", "self-gripping mesh", "coated mesh", "anti-adhesion barrier", "Seprafilm", "Interceed",
        "Parietex", "Proceed mesh", "Physiomesh", "Ventralex", "ProGrip", "DynaMesh", "GORE-TEX",
        "DualMesh", "Strattice", "Surgisis", "Permacol", "Phasix", "P4HB", "OviTex", "Vicryl mesh",
        "scaffold", "tissue engineering", "nanomaterial", "nanoparticle", "coating", "drug-eluting", "antimicrobial coating"
    ],
    "sensors_physics_magnetic_electromagnetic_pressure": [
        "Biosensing Techniques[Mesh]", "Electromagnetic Fields[Mesh]", "magnetic", "magnet", "magnets",
        "magnetically", "magnetic compression", "magnetic anastomosis", "magnetic sphincter", "magnetic sphincter augmentation",
        "magnetic resonance", "MRI", "magnetic navigation", "electromagnetic", "electromagnetic tracking",
        "electromagnetic field", "electric field", "electrical impedance", "impedance", "bioimpedance", "dielectric",
        "sensor", "sensors", "biosensor", "pressure sensor", "force sensor", "strain sensor", "tactile sensor",
        "haptic", "force feedback", "tactile feedback", "pressure", "manometry", "tensiometry", "tension",
        "flow sensor", "perfusion sensor", "oxygen sensor", "pH sensor", "temperature sensor", "thermal sensor",
        "accelerometer", "gyroscope", "inertial measurement", "RFID", "wireless", "Bluetooth", "wearable",
        "smart device", "smart instrument", "microfluidic", "lab-on-chip", "MEMS", "nanotechnology", "nanoparticle",
        "spectrometer", "spectroscopy", "optical fiber", "fiberoptic", "fiber-optic", "laser Doppler",
        "acoustic", "ultrasonic", "vibration", "elastography", "mechanical", "mechanobiology", "stiffness"
    ],
    "endoscopic_devices_stents_clips_vacuum": [
        "Endoscopy, Digestive System[Mesh]", "endoscopic", "endoscopy", "endoscope", "duodenoscope", "colonoscope",
        "gastroscope", "cholangioscope", "SpyGlass", "endoscopic ultrasound", "EUS", "ERCP", "sphincterotome",
        "guidewire", "catheter", "balloon catheter", "dilation balloon", "balloon dilatation", "papillary balloon",
        "stent", "self-expandable metal stent", "SEMS", "covered stent", "plastic stent", "biliary stent",
        "pancreatic stent", "lumen-apposing metal stent", "LAMS", "AXIOS", "Hot AXIOS", "Niti-S",
        "clip", "clips", "through-the-scope clip", "TTS clip", "over-the-scope clip", "OTSC", "Padlock clip",
        "endoscopic suturing", "OverStitch", "full-thickness resection device", "FTRD", "resection cap", "band ligation",
        "endoloop", "snare", "knife", "dual knife", "IT knife", "HookKnife", "HybridKnife", "Coagrasper",
        "hemostatic forceps", "endoscopic vacuum", "endoluminal vacuum", "EndoVAC", "sponge", "vacuum therapy",
        "nasocystic drain", "peroral cholangioscopy", "confocal laser endomicroscopy", "CLE", "endomicroscopy"
    ],
    "perfusion_ablation_interventional_oncology": [
        "ablation", "tumor ablation", "radiofrequency ablation", "RFA", "microwave ablation", "MWA",
        "cryoablation", "irreversible electroporation", "IRE", "NanoKnife", "laser ablation", "thermal ablation",
        "high-intensity focused ultrasound", "HIFU", "focused ultrasound", "embolization", "chemoembolization",
        "TACE", "radioembolization", "SIRT", "Y-90", "yttrium", "brachytherapy", "intraoperative radiotherapy",
        "IORT", "HIPEC", "hyperthermic intraperitoneal chemotherapy", "PIPAC", "pressurized intraperitoneal aerosol chemotherapy",
        "electrochemotherapy", "photodynamic therapy", "PDT", "perfusion", "isolated hepatic perfusion", "portal vein embolization",
        "hepatic venous deprivation", "radiological intervention", "image-guided ablation", "needle tracking", "navigation ablation"
    ],
}

# Optional disease enrichment with surgery terms, to catch disease-first records.
SURGICAL_QUALIFIER = [
    "surgery", "surgical", "operative", "operation", "procedure", "resection", "repair", "anastomosis",
    "laparoscopic", "robotic", "endoscopic", "minimally invasive"
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


def esearch_pmids(query: str) -> Tuple[int, List[str]]:
    """Return count and all PMIDs for one query, paging beyond 10k.

    Uses retmode=xml because NCBI occasionally returns malformed JSON for long
    complex queries with control characters inside the translated query string.
    """
    first = eutils_post("esearch.fcgi", {
        "db": DB, "term": query, "retmode": "xml", "retmax": "0", "usehistory": "n"
    })
    count, _ = parse_esearch_xml(first.content)
    pmids: List[str] = []
    for retstart in range(0, count, SEARCH_RETMAX):
        res = eutils_post("esearch.fcgi", {
            "db": DB, "term": query, "retmode": "xml",
            "retstart": str(retstart), "retmax": str(SEARCH_RETMAX), "usehistory": "n"
        })
        _, ids = parse_esearch_xml(res.content)
        pmids.extend(ids)
    return count, pmids


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
            t_block = or_block(TECH_MODULES[t_name])
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
