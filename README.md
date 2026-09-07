# Document & Identity Verification — Backend

FastAPI backend for the hackathon plan. Local/demo scope, not production.

## 1. Setup

```bash
# System deps (QR + OCR need native libraries)
sudo apt-get install -y libzbar0 tesseract-ocr        # Debian/Ubuntu
brew install zbar tesseract                            # macOS

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# UIDAI signing certificate — fetch from UIDAI, do not copy from a random repo
mkdir -p certs && cp ~/Downloads/uidai_signing.cer certs/

export CONSENT_SUBJECTS="vishu,personA,personB,personC,personD"
uvicorn app.main:app --reload --port 8000
```

Check `GET /health`. If `uidai_certificate_loaded` is `false`, signature
verification is skipped and every Aadhaar returns `UNVERIFIABLE`. Fix that
before the demo; it is the single point of failure for your showpiece.

Run `python tests/test_core.py` before demo day. It builds a synthetic signed
Secure QR payload, verifies it, tampers it, and asserts the signature breaks.

## 2. Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/verify/aadhaar` | QR decode, UIDAI signature verify, Verhoeff, ELA |
| POST | `/verify/aadhaar-full` | The above **plus** face-match against the signed QR photo |
| POST | `/verify/pan` | OCR, structural validation, ELA |
| POST | `/verify/marksheet` | OCR, arithmetic and range checks, baseline scan, ELA |
| POST | `/verify/face` | Standalone ID photo vs live frames |
| GET | `/dashboard/history` | Recent checks with reasons |
| GET | `/dashboard/summary` | Verdict counts, per-day trend, top reason codes |
| GET | `/reasons` | Full reason-code dictionary for the UI to render |
| GET | `/health` | Cert loaded, consent enforcement state |

## 3. How each check actually works

**Aadhaar Secure QR (V2).** The payload is a very long decimal string. Convert
to a big integer, then to bytes, gunzip, and you get 0xFF-delimited text fields,
a JPEG2000 photo, optional SHA-256 contact hashes, and a trailing 256-byte RSA
signature. That signature is SHA256-with-RSA over every preceding byte, made
with UIDAI's private key. Verify it against UIDAI's public certificate and you
have proved the field values are issuer-signed and bit-for-bit unaltered.

Two implementation traps, both handled in `aadhaar_qr.py`:

1. Do not `data.split(b"\xff")`. The embedded photo is binary and its JPEG
   markers contain 0xFF, so a naive split shreds the image and shifts every
   later offset. Scan for exactly the 16 delimiters you need and stop.
2. You cannot find where the photo ends by scanning forward, because its length
   is not declared. Count backwards from the signature instead: subtract 256 for
   the signature, then 32 bytes per contact hash indicated by field 0.

**Legacy V1 QR** is plain XML with no signature. Anyone can type it into a text
editor and generate a QR. The code returns `QR_V1_LEGACY_UNSIGNED` and the
verdict engine refuses to call it genuine. If your app ever shows a green tick
for a V1 QR, it is lying.

**Verhoeff checksum.** Dihedral-group-D5 check digit over the first 11 digits.
Catches every single-digit error and every adjacent transposition. It is pure
arithmetic on the number, so a forger generates a valid one in a `for` loop.
A pass means "not obviously garbage", never "real".

**PAN.** Format only: five letters, four digits, one letter, with the fourth
character encoding holder type. There is no public check digit and no offline
cryptographic material. The only authoritative check is the Income Tax
Department's own API.

**ELA.** JPEG is roughly idempotent, so re-saving an already-compressed region
changes it little, while a region with a different compression history changes
more. Re-encode at quality 90, diff, and flag blocks whose mean error is 2.5
sigma above the image's own distribution. Relative thresholding is deliberate:
an absolute one fails on both dim phone photos and clean scans.

**Marksheet.** Arithmetic consistency is the useful check. Subject marks that
do not sum to the printed total is a real, explainable catch, and it is what to
put on screen when a judge says "show it catching a fake, live." Range checks
and percentage recomputation back it up. I did **not** implement font-forensics
from your plan: glyph metrics are not recoverable from Tesseract output, and
per-character height variance tracks scan skew far more than tampering.

**Face-match and liveness.** ArcFace embeddings, cosine distance, threshold
0.68 (DeepFace's own tuned default for that model — do not reuse a threshold
across models). Liveness compares MediaPipe landmark geometry across a burst of
frames: eye-aspect-ratio dip for blink, normalised nose offset for head turn.

## 4. The verdict engine, and why it differs from your plan

Your table lists seven checks feeding one verified/flagged verdict. Averaging or
weighting them is wrong in both directions: a genuine document photographed
badly picks up ELA noise despite carrying a cryptographic proof of issuance, and
a forgery with a clean compression history gains points it never earned. A proof
and a heuristic do not belong in the same arithmetic. So checks are tiered and
the strongest available tier decides:

| Tier | Checks | Weight in verdict |
|---|---|---|
| 1 Cryptographic | UIDAI QR signature | Decides outright. Nothing below can overturn it. |
| 2 Structural | Verhoeff, PAN pattern, marksheet arithmetic | A failure is a hard flag. A pass means little. |
| 3 Heuristic | ELA, baseline alignment | Advisory only. Never the sole cause of a flag. |

Five verdicts, and none of them is the word "verified":

- `GENUINE_SIGNED` — Tier 1 passed. Issuer-signed, unaltered.
- `FORGED_SIGNATURE` — Tier 1 failed. Your strongest possible claim.
- `STRUCTURALLY_INVALID` — Tier 2 failed. Explainable and document-specific.
- `UNVERIFIABLE` — no Tier 1 available, Tier 2 clean. **This is the honest
  outcome for every PAN and every marksheet. It is not a pass.**
- `NEEDS_REVIEW` — Tier 3 raised something on an otherwise clean document.

`identity_binding` is reported **separately** from the verdict, on its own axis:
`BOUND`, `NOT_BOUND`, `LIVENESS_FAILED`, `NOT_ATTEMPTED`. Keep them separate in
the UI. A genuine Aadhaar held by the wrong person is `GENUINE_SIGNED` +
`NOT_BOUND`, and collapsing those into one badge is precisely how the "can't a
forger copy a real QR?" question sinks a demo.

## 5. Privacy constraints, enforced in code rather than in policy

- **Biometrics.** Frames live in local variables, temp files are zero-filled and
  `fsync`ed before unlink in a `finally` block, and no embedding or crop reaches
  the response or the database.
- **Audit schema.** `store.py` has no image column, no embedding column, no full
  identifier column. A column you never populate eventually gets populated by
  someone at 3am. A column that does not exist does not.
- **Consent.** `/verify/face` and `/verify/aadhaar-full` return 403 unless
  `consent_subject` is in `CONSENT_SUBJECTS`. The whiteboard rule "only
  consenting teammates" survives contact with a 2am test session this way and
  not otherwise.
- **Redaction.** `reference_id` is masked server-side, so a frontend bug cannot
  leak a real number into a screenshot.
- **Disclaimer** ships inside every response body. The frontend has to actively
  discard it to mislead someone.

## 6. Judge questions, with answers I would actually give

**"Is this real verification or just format checking?"** Both, and the app says
which. Aadhaar is real cryptographic verification against UIDAI's signature.
PAN and marksheets are consistency checking only, and they return
`UNVERIFIABLE`, never a pass. Naming that boundary yourself is worth more than
any demo polish.

**"Can't a forger just copy real numbers or QR codes?"** Yes, and that attack
defeats signature verification completely, because a copied QR is genuinely
signed. It is why `/verify/aadhaar-full` compares the live face against the
photo *inside the signed QR payload* rather than a crop off the card face. The
signature proves the document was issued; the face-match proves it is yours.
Two properties, two mechanisms.

**"What's your accuracy, and on what data?"** Signature verification is not
statistical, so accuracy is not the right frame for it: it is a cryptographic
check that passes or fails. Face-match uses ArcFace's published benchmarks, not
ours. ELA we have not measured on a labelled dataset, so we ship it as a
reviewer heatmap and not as a classifier. That answer is stronger than an
invented percentage, and an invented percentage is the thing most likely to get
you taken apart in Q&A.

**"Do you have legal permission to process Aadhaar data?"** Offline QR
verification against UIDAI's published certificate is the intended offline path
and needs no database access. Be precise that you are not a registered
AUA/KUA, you query nothing, you store nothing, and demo data came from
teammates who were asked directly. Get someone qualified to confirm the
specifics before you present, since I am not the right source for that.

**"How does this scale beyond a demo?"** Signature verification is a few
milliseconds of RSA and scales trivially. The real costs are face-match compute
and, for genuine PAN or board verification, per-call API fees plus registration
you do not currently have.

## 7. Two things I would change about the priority order

Your fallback order is QR > Verhoeff > ELA > face-match > dashboard.

1. **Move face-match above ELA.** ELA is the weakest signal here and the most
   likely to misfire on stage. In testing against a synthetic edit it flagged
   printed-text edges more strongly than the actual patched region, which is the
   documented failure mode: sharp edges always show high error, and on an ID
   card that is most of the image. Face-match is what answers your hardest judge
   question. Trading it away for a heatmap that highlights text is a bad swap.
2. **Verhoeff is nearly free** — one file, no dependencies, tested above. Treat
   it as done rather than as a ranked priority, and spend the slot elsewhere.
