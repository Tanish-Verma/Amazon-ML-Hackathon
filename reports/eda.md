# Phase 1 — EDA

## 1. Ground-truth shape

* 2,206,821 Source-1 entities; **123,247 singletons (5.58%)**
* 7,638,365 matched IDs total; mean 3.67 per matched entity
* max list size 11

| List size | Entities |
|---|---|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

## 2. Distractors (records matching nothing)

* Source 2: 1,340,997 of 5,034,616 (26.6%) match no Source-1 entity
* Source 3: 1,340,857 of 5,285,603 (25.4%)

These are the precision hazard: they sit in the candidate pool looking plausible,
and every one we accept costs a full point on a singleton or dilutes a matched entity.

## 3. Writing systems

**Source 1 (train)**

| Country | Script mix (names) |
|---|---|
| India | Latin 100.0% |
| US | Latin 100.0% |

**Source 2 (train)**

| Country | Script mix (names) |
|---|---|
| India | Latin 77.1%, Devanagari 13.0%, Telugu 1.9%, Kannada 1.8%, Tamil 1.6% |
| US | Latin 100.0%, None 0.0% |

**Source 3 (train)**

| Country | Script mix (names) |
|---|---|
| India | Latin 87.9%, Devanagari 6.8%, Telugu 1.0%, Kannada 1.0%, Tamil 0.9% |
| US | Latin 100.0%, None 0.0% |

## 4. Name-token document frequency, per country

Basis for deriving stopwords from corpus statistics rather than a hand-written
per-country table.

| Country | Most frequent name tokens (document frequency %) |
|---|---|
| India | `limited` 59.1, `private` 48.9, `ltd` 16.5, `pvt` 13.8, `india` 6.7, `llp` 4.4, `services` 2.7, `solutions` 2.4, `brothers` 2.3, `trading` 2.3, `co` 2.2, `technologies` 1.9, `international` 1.7, `foundation` 1.6, `global` 1.5, `industries` 1.5, `tech` 1.5, `enterprises` 1.4 |
| US | `llc` 26.9, `inc` 18.0, `and` 4.3, `s` 4.2, `c` 4.0, `care` 3.2, `of` 2.8, `associates` 2.7, `center` 2.4, `group` 2.4, `partners` 2.2, `p` 2.1, `corp` 2.1, `l` 2.1, `pc` 2.0, `health` 2.0, `d` 1.7, `clinic` 1.6 |

### Test split, including the unseen country (Source 1)

This is the direct test of whether derived stoplists generalise to a country
absent from training. It needs no labels -- it is a frequency count -- so it can
legitimately be computed on the test split.

| Country | Most frequent name tokens (document frequency %) |
|---|---|
| France | `sarl` 28.3, `sas` 20.1, `club` 8.7, `france` 8.0, `de` 6.7, `eurl` 6.5, `ecole` 6.0, `amicale` 5.5, `comite` 5.3, `sa` 4.9, `maison` 4.2, `du` 4.1, `sasu` 4.1, `centre` 4.1, `union` 3.3, `sci` 3.2, `sportive` 3.2, `des` 3.0 |
| US | `llc` 26.9, `inc` 18.0, `and` 4.3, `s` 4.2, `c` 3.9, `care` 3.2, `of` 2.8, `associates` 2.8, `center` 2.4, `group` 2.4, `partners` 2.2, `corp` 2.1, `p` 2.1, `l` 2.0, `pc` 2.0, `health` 2.0, `d` 1.7, `clinic` 1.7 |
| India | `limited` 59.1, `private` 48.9, `ltd` 16.5, `pvt` 13.7, `india` 6.7, `llp` 4.4, `services` 2.7, `solutions` 2.4, `trading` 2.3, `brothers` 2.3, `co` 2.2, `technologies` 1.9, `international` 1.8, `foundation` 1.6, `global` 1.6, `industries` 1.5, `tech` 1.5, `enterprises` 1.4 |

**Confirmed.** France's `sarl` (28.3%) and `sas` (20.1%) sit in the same
frequency band as US `llc` (26.9%) and `inc` (18.0%). A document-frequency
threshold down-weights them automatically, with no French-specific table and no
code change.

Three things fall out of the same mechanism that a hand-written list would have
missed:

* **French function words** -- `de` 6.7%, `du` 4.1%, `des` 3.0%. A hardcoded
  *English* stopword list would not have touched these.
* **Generic category words** -- `club` 8.7%, `ecole` 6.0%, `amicale` 5.5%,
  `comite` 5.3%, `maison` 4.2%, mirroring US `care`/`center`/`group`.
* **The country's own name** -- `france` 8.0%, mirroring India's `india` 6.7%.

## 5. Lexical reachability of true matches  **(the headline)**

Sample: 150,000 matched Source-1 entities, **549,843 true pairs**.
A pair counts as reachable if normalised name char-3gram Jaccard >= 0.15
OR address token Jaccard >= 0.2 OR the addresses share a digit run.
Thresholds are deliberately generous: this is an optimistic ceiling for a purely
lexical pipeline, so whatever falls outside is genuinely hard.

| Channel | Pairs | % of true matches |
|---|---|---|
| Name reachable | 497,239 | 90.43% |
| Address reachable | 521,371 | 94.82% |
| **Either (lexical ceiling)** | **549,163** | **99.88%** |
| Address rescues a hopeless name | 51,924 | 9.44% |
| Name rescues a hopeless address | 27,792 | 5.05% |
| **Neither — needs embeddings** | **680** | **0.12%** |
| (of which, address was empty) | 24,519 | 4.46% |

### By country

| Country | Pairs | Lexically reachable |
|---|---|---|
| India | 219,748 | 99.77% |
| US | 330,095 | 99.94% |

### By writing system of the matched record's name

| Script | Pairs | Name channel alone | Either channel |
|---|---|---|---|
| Latin | 512,095 | 97.1% | 99.9% |
| Devanagari | 21,428 | 0.6% | 99.1% |
| Telugu | 3,160 | 0.5% | 98.9% |
| Kannada | 2,952 | 0.6% | 98.1% |
| Tamil | 2,695 | 0.5% | 98.8% |
| Gujarati | 2,458 | 0.3% | 98.5% |
| Bengali | 2,390 | 0.9% | 99.1% |
| Malayalam | 1,511 | 0.6% | 99.0% |
| Oriya | 594 | 1.0% | 99.3% |
| Gurmukhi | 526 | 0.2% | 99.4% |

### Examples of unreachable pairs

```
[India] S1-255934287  'Guru Infra Private Limited'
          'House No. 671, C/O Memon Afaque, Sobani Manzil, Gawalipura, Sadar, Nagpur, Maharashtra'
     -> S2-554564956  'गुरु इंफ्रा प्राइवेट लिमिटेड'
          'HOUSE NO. 6-71, NAGPUR, महाराष्ट्र'

[India] S1-34564838  'Good Industries Private Limited'
          '48, 1St Floor, New Modella Co-Op Premises Soc Ltd Padwal Ngr, Panchpakhadi, Wagle Indl. Es, Tate, Thane, Maharashtra'
     -> S2-171372315  'गुड इंडस्ट्रीज प्राइवेट लिमिटेड'
          '4-8, THANE, Maharashtra'

[India] S1-699757113  'Aditya Builders Private Limited'
          'Tamil Nadu, New No 306, Anna Salai, Thousand Light, Old No 669, Chennai'
     -> S2-129838756  'ஆதித்யா பில்டர்ஸ் பிரைவேட் லிமிடெட்'
          'OLD NO 69, CHENNAI CITY REGION, தமிழ்நாடு'

[India] S1-59232116  'First Technology Pvt Ltd'
          'Tower C, Flat No. 1602, Bestech Park View Spa, Sector 47, Near Dps P, Gurgaon, Haryana'
     -> S3-440044571  'फर्स्ट टेक्नोलॉजी प्रा. लि.'
          'Tower C, Gurugram, Gurgaon, HR'

[India] S1-541400546  'Premier & Co'
          '#1048, 20Th Main Road 5Th Block, Rajajinagar, Bangalore, Karnataka'
     -> S3-657566433  'Phrbgr +  Co'
          'No 048, Bangalore, KA'

[India] S1-423949339  'Shakti It Private Limited'
          'Swastik House, 3Rd Floor, M.N. Marg, L J Road, Mahim, Mumbai, Mumbai City, Maharashtra'
     -> S3-596169111  'शक्ति आईटी प्राइवेट लिमिटेड'
          'Swastik House, null, Mumbai, Mumbai Ii, MH'

[US] S1-264568202  'Great Network'
          '7210 Imbrie Drive, Fl 1, Hillsboro, OR'
     -> S3-374337417  'Great Sérvices Enterprises'
          ''

[India] S1-206270853  'Big Engineering Private Limited'
          '108, Zulfiqar Ganj, Shaym Ganj, Bareilly, Uttar Pradesh'
     -> S3-419975895  'बिग इंजीनियरिंग प्राइवेट लिमिटेड'
          '00108, Bareilly, उत्तर प्रदेश'

[India] S1-593070103  'Smart It Pvt Ltd'
          '54, 1St A Main, 2Nd Stage, 1St Block Raj Mahal Vilas Extension, Bangalore, Karnataka'
     -> S3-718065408  'ಸ್ಮಾರ್ಟ್ ಐಟಿ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್'
          '55, Bangalore, KA'

[India] S1-119906168  'QRL Media LLP'
          'C/O Sanjeev Kumar, H.No 148 Jhapaha, Tola- Jhapaha, Block- Musahri, Muzaffarpur, Bihar'
     -> S2-646270923  'QRL LLP Center'
          ''

[US] S1-163540861  'RG Services L.L.C.'
          '231 Alexander Boulevard, Clarksville, TN'
     -> S3-26499424  'RG  L.L.C. Center'
          ''

[US] S1-652532913  'Donet Partnership L.L.C.'
          '702 Marian Court, Merced, CA'
     -> S3-825865657  'Donet L.L.C. Cénter | www.donetll.com'
          'Marian Ct, Mered, California'

```

---

## 6. What this means for the plan

### Headline: embeddings are not needed for recall

A purely lexical pipeline reaches **99.88%** of all true matches. Only **0.12%**
(680 of 549,843 sampled pairs) are invisible to both a character-n-gram name index and a
token/digit address index. Phase 1 existed to decide whether the embedding channel was
load-bearing. It is not.

Since F0.5 weights precision 2x over recall, chasing a final 0.12% of recall cannot repay the GPU
hours, the storage, or the engineering time -- all of which are better spent on precision.

### The address channel solves transliteration, not embeddings

For every Indic script in the data, the **name channel alone reaches 0.2-1.0%** -- effectively
zero, exactly as expected when a transliterated name shares no characters with its Latin form. But
the **union reaches 98.1-99.4%**, because addresses keep their digits and largely Latin place
names, and street numbers survive transliteration untouched.

### Both channels are mandatory

* Name alone 90.43%; address alone 94.82%; union 99.88%.
* The address rescues **9.44%** of pairs whose names are hopeless.
* The name rescues **5.05%** of pairs whose addresses are unusable (4.46% of matched records have
  no address at all).

### The real problem is precision

26.6% of Source-2 and 25.4% of Source-3 records match **nothing** -- roughly 2.7M distractors in
the pool. With a 5.58% singleton rate where one false merge costs a full point, this is where the
score is won or lost.

### Proposed change to the approved plan

**Drop the embedding channel from Phase 3 blocking.** It buys at most 0.12% recall.

**Keep one embedding option open as a Phase 4 *matching feature*.** The corrected by-script numbers
strengthen this case rather than weaken it: for a transliterated record retrieved by address, every
lexical name feature is ~0 *whether or not the match is real*, so the classifier is blind exactly
where several candidates share an address. A multilingual embedding cosine would restore signal
there. It is cheap because it only embeds the capped candidate set, not the 10M-record corpus, and
it is deferrable: it changes the feature matrix only, leaving `candidate_pairs.tsv` untouched.

## 7. Correction: a bug in this analysis, and what it teaches Phase 2

The first run of this EDA used `[^\w\s]` to strip punctuation. That looks correct and is wrong for
Indic scripts: their vowel signs are Unicode categories `Mn`/`Mc`, which `\w` does not match, so
`प्राइवेट` was shattered into `प र इव ट`. It surfaced as single Devanagari characters topping the
India document-frequency table.

Fixed by stripping on Unicode *category* (`P`/`S`/`C`) instead of `\w`. The headline moved only
from 99.86% to 99.88%, because the bug degraded only the name channel for Indic scripts while the
address channel -- which is what actually rescues those pairs -- was unaffected. The original figure
was a lower bound, not an overstatement.

**Phase 2 must not repeat this.** Script-agnostic tokenisation means neither `[a-z]+` nor `\w`;
it means preserving categories `L*`, `N*` and `M*` and stripping only `P*`/`S*`/`C*`. This is a
worked example of the failure mode, already observed in our own data.
