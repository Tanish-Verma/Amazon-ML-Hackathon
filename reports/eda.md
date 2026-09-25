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

## 5. Lexical reachability of true matches  **(the headline)**

Sample: 150,000 matched Source-1 entities, **549,843 true pairs**.
A pair counts as reachable if normalised name char-3gram Jaccard >= 0.15
OR address token Jaccard >= 0.2 OR the addresses share a digit run.
Thresholds are deliberately generous: this is an optimistic ceiling for a purely
lexical pipeline, so whatever falls outside is genuinely hard.

| Channel | Pairs | % of true matches |
|---|---|---|
| Name reachable | 497,392 | 90.46% |
| Address reachable | 521,142 | 94.78% |
| **Either (lexical ceiling)** | **549,096** | **99.86%** |
| Address rescues a hopeless name | 51,704 | 9.40% |
| Name rescues a hopeless address | 27,954 | 5.08% |
| **Neither — needs embeddings** | **747** | **0.14%** |
| (of which, address was empty) | 24,519 | 4.46% |

### By country

| Country | Pairs | Lexically reachable |
|---|---|---|
| India | 219,748 | 99.74% |
| US | 330,095 | 99.94% |

### By writing system of the matched record's name

| Script | Pairs | Name channel alone | Either channel |
|---|---|---|---|
| Latin | 512,095 | 97.1% | 99.9% |
| Devanagari | 21,428 | 0.9% | 98.9% |
| Telugu | 3,160 | 0.9% | 98.8% |
| Kannada | 2,952 | 1.2% | 98.0% |
| Tamil | 2,695 | 0.9% | 98.7% |
| Gujarati | 2,458 | 0.6% | 98.3% |
| Bengali | 2,390 | 1.5% | 98.7% |
| Malayalam | 1,511 | 0.8% | 98.9% |
| Oriya | 594 | 1.3% | 99.0% |
| Gurmukhi | 526 | 0.8% | 99.4% |

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

[India] S1-210028882  'Essel & Brothers Limited'
          'Village Jani Chak, Taragarh To Jhabkara(Marara) Road, Tehsil, Pathankot, Gurdaspur, Punjab'
     -> S2-962123448  'Avipyra'
          'VILALGE JANI CHAK, GURDASPUR, ਪੰਜਾਬ'

[India] S1-119906168  'QRL Media LLP'
          'C/O Sanjeev Kumar, H.No 148 Jhapaha, Tola- Jhapaha, Block- Musahri, Muzaffarpur, Bihar'
     -> S2-646270923  'QRL LLP Center'
          ''

[US] S1-163540861  'RG Services L.L.C.'
          '231 Alexander Boulevard, Clarksville, TN'
     -> S3-26499424  'RG  L.L.C. Center'
          ''

```

---

## 6. What this means for the plan

### The headline finding: embeddings are not needed for recall

A purely lexical pipeline can reach **99.86%** of all true matches. Only **0.14%**
(747 of 549,843 sampled pairs) are invisible to both a character-n-gram name index and a
token/digit address index. Phase 1 existed to decide whether the embedding channel was
load-bearing or optional. The answer is unambiguous: **optional, and barely that.**

Since F0.5 weights precision 2x over recall, chasing a final 0.14% of recall is close to
worthless — it cannot plausibly repay the GPU hours, the 7.7 GB of embedding storage, or the
engineering time, all of which are better spent on precision.

### Why the transliterations are already solved

This was the risk that motivated the embedding plan, and the by-script table explains why it
evaporates. For Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Oriya and
Gurmukhi names, the **name channel alone reaches ~1%** — exactly as expected, since a
transliterated name shares no characters with its Latin form. But the **union reaches 98-99%**,
because the address is still written with digits and largely Latin place names. Street numbers in
particular survive transliteration untouched.

The address channel, not an embedding model, is what solves transliteration.

### Both channels are mandatory

Neither channel is sufficient alone, and the plan's insistence on an independent union is
vindicated by hard numbers:

* Name alone: 90.46%. Address alone: 94.78%. Union: 99.86%.
* The address rescues **9.40%** of pairs whose names are hopeless (the transliterations).
* The name rescues **5.08%** of pairs whose addresses are empty or useless (4.46% of all matched
  records have no address at all).

### Derived stopwords will generalise to France

The DF profile confirms the hypothesis behind the derived-statistics design. The dominant name
tokens are legal-form boilerplate sitting in a very high frequency band: India `limited` 59.1%,
`private` 48.9%, `ltd` 16.5%, `pvt` 13.8%; US `llc` 26.9%, `inc` 18.0%. France's `SARL`, `SAS`,
`SASU` and `EURL` will land in the same band by the same mechanism, so a DF-threshold rule
down-weights them with no French-specific table and no code change.

### The real problem is precision, not recall

26.6% of Source-2 and 25.4% of Source-3 records match **nothing**. Roughly 2.7M distractors sit in
the pool looking plausible. Combined with a 5.58% singleton rate where a single false merge costs
a full point, this is where the score will actually be won or lost.

### Proposed change to the approved plan

**Drop the embedding channel (channel D) from Phase 3 blocking.** It buys at most 0.14% recall for
hours of GPU time and gigabytes of storage.

**But keep one embedding option open as a *matching feature*, not a blocking channel.** For an
Indic-script record pulled in by address, the lexical name features are all ~0 whether or not the
match is real, so the classifier is partly blind exactly where several candidates share an address.
A multilingual embedding cosine would restore signal there. That is a *precision* argument, and it
applies to a meaningful slice: India is 47% of test Source-1 entities, and ~13% of India's
Source-2 names are non-Latin. Worth testing in Phase 4 on measured feature importance — cheap,
because it only has to embed the capped candidate set rather than the full 10M-record corpus.
