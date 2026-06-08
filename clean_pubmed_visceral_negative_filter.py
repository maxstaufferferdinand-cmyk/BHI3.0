from pathlib import Path
import pandas as pd
import re

# ==========================================
# SETTINGS
# ==========================================
INPUT_CSV = Path(r"C:\Users\Max Stauffer\BreakingRules\pubmed_visceral_surgery_technology_precise_1980_2025_v7_no_broad_duplicates.csv")
OUTPUT_BASENAME = "pubmed_visceral_surgery_technology_negative_clean_1980_2025"
OUTPUT_DIR = INPUT_CSV.parent

YEAR_MIN = 1980
YEAR_MAX = 2025
CHUNKSIZE = 25000

# If True: exclude a record only if exclusion term is found AND no abdominal/visceral rescue term is found.
# This prevents losing e.g. esophagectomy + thoracoscopy, vascular resection in pancreatic cancer, etc.
USE_RESCUE_TERMS = True

# Strong negative exclusion terms: mostly isolated non-visceral fields.
# Keep these specific enough to avoid deleting relevant abdominal records.
NEGATIVE_PATTERNS = {
    "neurosurgery_neurology": [
        r"\bneurosurg\w*\b", r"\bbrain\b", r"\bcerebral\b", r"\bcerebrovascular\b", r"\bintracranial\b",
        r"\bsubarachnoid\b", r"\bsubdural\b", r"\bepidural hematoma\b", r"\bspinal cord\b", r"\bspine surgery\b",
        r"\bventriculoperitoneal shunt\b", r"\bventriculoatrial shunt\b", r"\bhydrocephalus\b", r"\bcraniotom\w*\b",
        r"\bglioma\b", r"\bmeningioma\b", r"\bpituitary\b", r"\baneurysm clipping\b"
    ],
    "orthopedics_trauma": [
        r"\borthop\w*\b", r"\barthroplast\w*\b", r"\bhip replacement\b", r"\bknee replacement\b",
        r"\btotal hip\b", r"\btotal knee\b", r"\bfracture fixation\b", r"\bintramedullary nail\w*\b",
        r"\bspinal fusion\b", r"\bpedicle screw\w*\b", r"\bosteosynthesis\b", r"\bacetabular\b",
        r"\bmeniscus\b", r"\brotator cuff\b", r"\bshoulder arthroscop\w*\b", r"\bACL reconstruction\b"
    ],
    "cardiac_vascular_non_visceral": [
        r"\bcardiac surg\w*\b", r"\bcardiopulmonary bypass\b", r"\bcoronary artery bypass\b", r"\bCABG\b",
        r"\bheart valve\b", r"\bmitral valve\b", r"\baortic valve\b", r"\bventricular assist\b",
        r"\bperipheral arterial disease\b", r"\bfemoral arter\w*\b", r"\bpopliteal arter\w*\b", r"\btibial arter\w*\b",
        r"\bcarotid endarterectomy\b", r"\bcarotid stent\w*\b", r"\bendovascular aneurysm repair\b", r"\bEVAR\b",
        r"\babdominal aortic aneurysm\b", r"\baortic aneurysm\b", r"\bangioplasty\b", r"\bvascular graft\b"
    ],
    "thoracic_non_esophageal": [
        r"\bthoracic surg\w*\b", r"\blung resection\b", r"\blobectomy\b", r"\bpneumonectomy\b", r"\bsegmentectomy of the lung\b",
        r"\bVATS\b", r"\bvideo-assisted thoracoscopic\b", r"\bpleurodesis\b", r"\bpulmonary nodule\b",
        r"\bbronchoscopy\b", r"\btrachea\b", r"\btracheal\b", r"\bmediastinal\b", r"\bthymectomy\b"
    ],
    "urology_gynecology_isolated": [
        r"\burolog\w*\b", r"\bprostatectom\w*\b", r"\bnephrectom\w*\b", r"\bcystectom\w*\b", r"\bTURP\b",
        r"\bureteroscop\w*\b", r"\bpercutaneous nephrolithotom\w*\b",
        r"\bgynecolog\w*\b", r"\bhysterectom\w*\b", r"\boophorectom\w*\b", r"\bmyomectom\w*\b", r"\bendometriosis\b"
    ],
    "ent_ophthalmology_dental": [
        r"\botolaryngolog\w*\b", r"\bENT\b", r"\btonsillectom\w*\b", r"\bsinus surg\w*\b", r"\brhinolog\w*\b",
        r"\bophthalmolog\w*\b", r"\bcataract\b", r"\bvitrectom\w*\b", r"\bretinal\b",
        r"\bdental\b", r"\borthodont\w*\b", r"\bmaxillofacial\b", r"\bmandibular\b"
    ],
    "anesthesia_only": [
        r"\banesthesia\b", r"\banaesthesia\b", r"\banesthetic\b", r"\banaesthetic\b", r"\bhalothane\b", r"\benflurane\b",
        r"\bisoflurane\b", r"\bsevoflurane\b", r"\bdesflurane\b", r"\bpropofol\b", r"\bneuromuscular block\w*\b"
    ],
}

# Rescue terms: if one of these appears, we keep the record even if a negative term appears.
# This is deliberately broad for abdominal/visceral relevance.
RESCUE_PATTERNS = [
    r"\babdom\w*\b", r"\bvisceral\b", r"\bgastrointestin\w*\b", r"\bGI\b", r"\bdigestive\b",
    r"\besophag\w*\b", r"\boesophag\w*\b", r"\bstomach\b", r"\bgastr\w*\b", r"\bduoden\w*\b",
    r"\bsmall bowel\b", r"\bsmall intestine\b", r"\bileum\b", r"\bjejunal\b", r"\bjejunum\b",
    r"\bcolon\b", r"\bcolonic\b", r"\bcolorectal\b", r"\brectal\b", r"\brectum\b", r"\banal\b", r"\banorectal\b",
    r"\bliver\b", r"\bhepatic\b", r"\bhepatectom\w*\b", r"\bpancrea\w*\b", r"\bpancreatectom\w*\b",
    r"\bbile duct\b", r"\bbiliary\b", r"\bcholangi\w*\b", r"\bgallbladder\b", r"\bcholecyst\w*\b",
    r"\bappend\w*\b", r"\bperitone\w*\b", r"\bmesenter\w*\b", r"\bomental\b", r"\bomentum\b",
    r"\bhernia\b", r"\binguinal\b", r"\bventral hernia\b", r"\bhiatal hernia\b", r"\bparaesophageal\b",
    r"\bfundoplication\b", r"\bNissen\b", r"\bToupet\b", r"\bDor\b", r"\bHeller myotomy\b", r"\bPOEM\b",
    r"\banastomos\w*\b", r"\bstapler\b", r"\bstapling\b", r"\bbariatric\b", r"\bgastric bypass\b", r"\bsleeve gastrectomy\b",
    r"\bCrohn\b", r"\bulcerative colitis\b", r"\bIBD\b", r"\bdiverticul\w*\b", r"\bcolitis\b",
    r"\bappendicitis\b", r"\bcholecystitis\b", r"\bpancreatitis\b", r"\bHCC\b", r"\bhepatocellular\b",
    r"\bcholangiocarcinoma\b", r"\bmetastatic colorectal\b", r"\bperitoneal carcinomatosis\b"
]

TEXT_COLUMNS = ["title", "abstract", "journal", "mesh_terms", "keywords", "publication_types"]


def compile_patterns(patterns):
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]

NEGATIVE_REGEX = {k: compile_patterns(v) for k, v in NEGATIVE_PATTERNS.items()}
RESCUE_REGEX = compile_patterns(RESCUE_PATTERNS)


def build_text(row):
    vals = []
    for c in TEXT_COLUMNS:
        if c in row and pd.notna(row[c]):
            vals.append(str(row[c]))
    return " | ".join(vals)


def find_negative_categories(text):
    cats = []
    for cat, regs in NEGATIVE_REGEX.items():
        if any(r.search(text) for r in regs):
            cats.append(cat)
    return cats


def has_rescue(text):
    return any(r.search(text) for r in RESCUE_REGEX)


def process_chunk(chunk):
    # hard year filter
    if "year" in chunk.columns:
        chunk["year_num"] = pd.to_numeric(chunk["year"], errors="coerce")
        chunk = chunk[(chunk["year_num"] >= YEAR_MIN) & (chunk["year_num"] <= YEAR_MAX)].copy()
        chunk = chunk.drop(columns=["year_num"], errors="ignore")

    if chunk.empty:
        return chunk, pd.DataFrame()

    texts = chunk.apply(build_text, axis=1)
    neg_cats = texts.apply(find_negative_categories)
    rescue = texts.apply(has_rescue)

    chunk["negative_filter_categories"] = neg_cats.apply(lambda x: ";".join(x))
    chunk["has_visceral_rescue_term"] = rescue

    # Exclude if negative category present and, if enabled, no rescue term.
    if USE_RESCUE_TERMS:
        exclude_mask = (chunk["negative_filter_categories"] != "") & (~chunk["has_visceral_rescue_term"])
    else:
        exclude_mask = (chunk["negative_filter_categories"] != "")

    excluded = chunk.loc[exclude_mask].copy()
    kept = chunk.loc[~exclude_mask].copy()

    return kept, excluded


def main():
    print("Input:", INPUT_CSV)
    print("Output dir:", OUTPUT_DIR)
    print("Use rescue terms:", USE_RESCUE_TERMS)

    kept_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.csv"
    excluded_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}_excluded.csv"
    summary_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}_summary.txt"

    # Remove previous outputs if rerunning
    for p in [kept_path, excluded_path, summary_path]:
        if p.exists():
            p.unlink()

    total_in = total_kept = total_excluded = 0
    category_counts = {cat: 0 for cat in NEGATIVE_PATTERNS}

    first_kept = True
    first_excl = True

    for i, chunk in enumerate(pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE, encoding="utf-8-sig", low_memory=False), start=1):
        total_in += len(chunk)
        kept, excluded = process_chunk(chunk)
        total_kept += len(kept)
        total_excluded += len(excluded)

        if len(excluded):
            for cats in excluded["negative_filter_categories"].dropna().astype(str):
                for cat in cats.split(";"):
                    if cat:
                        category_counts[cat] = category_counts.get(cat, 0) + 1

        kept.to_csv(kept_path, mode="a", index=False, header=first_kept, encoding="utf-8-sig")
        first_kept = False

        if len(excluded):
            excluded.to_csv(excluded_path, mode="a", index=False, header=first_excl, encoding="utf-8-sig")
            first_excl = False

        if i % 5 == 0:
            print(f"Chunk {i}: total_in={total_in:,}, kept={total_kept:,}, excluded={total_excluded:,}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Input CSV: {INPUT_CSV}\n")
        f.write(f"Total input rows parsed by pandas: {total_in}\n")
        f.write(f"Total kept rows: {total_kept}\n")
        f.write(f"Total excluded rows: {total_excluded}\n")
        f.write(f"Use rescue terms: {USE_RESCUE_TERMS}\n\n")
        f.write("Excluded category counts:\n")
        for cat, n in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{cat}: {n}\n")

    print("\nDONE")
    print("Kept:", kept_path)
    print("Excluded:", excluded_path)
    print("Summary:", summary_path)
    print(f"Rows: input={total_in:,}, kept={total_kept:,}, excluded={total_excluded:,}")


if __name__ == "__main__":
    main()
