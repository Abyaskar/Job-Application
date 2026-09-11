# Job Application Strategy AI

> **Don't just find jobs. Know which jobs to look at first --- and
> understand why.**

Job Application Strategy AI is a job-recommendation application that
compares a candidate's resume and career goal with available jobs.

The important difference is that the system does **not** treat one
similarity number as the candidate's "qualification percentage".

Instead, it combines several separate signals:

-   career intent
-   semantic similarity between resume and job description
-   required-skill coverage
-   experience
-   education
-   location
-   job requirements

It then applies an eligibility rule and ranks the jobs that remain.

The project also contains an **optional learned ranking experiment**
using Logistic Regression. The learned model is **not the default
ranking path**.

------------------------------------------------------------------------

# 1. What this project actually does

The simplest description is:

``` text
Resume
   +
Career goal
   +
Job information
   ↓
Compare the candidate with jobs
   ↓
Check required-skill coverage
   ↓
Rank the better matches
   ↓
Explain why
   ↓
Show what skills are missing
```

The application is trying to answer:

> **"Which jobs should this candidate consider first, based on the
> information currently available to the system?"**

It is **not** trying to predict:

> "Will the company hire this person?"

It is also **not** an ATS score.

------------------------------------------------------------------------

# 2. What the system does NOT claim

These distinctions are important.

## A semantic score of 75% does NOT mean:

``` text
75% qualified
```

## A final score of 82% does NOT mean:

``` text
82% chance of getting hired
```

## A model AUC of 0.90 does NOT mean:

``` text
90% hiring probability
```

## "Eligible" does NOT mean:

``` text
The company will accept the candidate
```

It means:

> **The candidate passed the current system's required-skill coverage
> rule.**

------------------------------------------------------------------------

# 3. What happens when a user uses the application?

A normal user flow is:

``` text
1. Upload resume
        ↓
2. Resume information is extracted
        ↓
3. Enter a career goal
        ↓
4. Career goal is normalized
        ↓
5. Jobs are retrieved
        ↓
6. Candidate/job signals are calculated
        ↓
7. Eligibility is checked
        ↓
8. Jobs are ranked
        ↓
9. Skill gaps and explanation are generated
        ↓
10. User decides what to do
```

The application can also keep recommendation results in cache so
repeated requests do not always have to perform the complete calculation
again.

------------------------------------------------------------------------

# 4. Resume understanding

The resume extraction layer creates a structured candidate profile.

Depending on what can be extracted from the document, the system works
with information such as:

-   skills
-   experience years
-   education
-   resume text
-   career/domain information

The quality of the recommendation depends on the quality of this
extraction.

For example:

If the resume clearly contains:

``` text
Python
```

but the extractor fails to detect it, the ranking system may incorrectly
think Python is missing.

So:

> **The recommendation system cannot be better than the information it
> receives from the resume and job description.**

------------------------------------------------------------------------

# 5. Career intent

A resume describes what a person has already done.

A career goal describes what the person wants to do next.

For example:

``` text
Resume:
Data Analyst

Career goal:
Generative AI Engineer
```

The system keeps these concepts separate.

This is important because otherwise a resume containing many
data-analysis keywords could keep pushing the candidate toward Data
Analyst jobs even when the candidate wants to move toward AI.

------------------------------------------------------------------------

# 6. How career intent is normalized

The current intent system uses a deterministic cascade:

``` text
1. Exact alias
       ↓
2. Fuzzy matching
       ↓
3. Keyword overlap
       ↓
4. Embedding fallback
```

For example:

``` text
"genai engineer"
```

can map to a canonical role.

A small spelling mistake such as:

``` text
"Data Analist"
```

can be handled by fuzzy matching.

A more descriptive goal such as:

``` text
"I want to work with LLMs and RAG"
```

can be resolved through relevant keywords.

The normal intent flow is therefore **not dependent on an LLM**.

------------------------------------------------------------------------

# 7. Important: intent is not just another embedding score

The current ranking code calculates intent alignment using deterministic
title/domain/intent matching.

The basic idea is:

``` text
Strong title/canonical match
    → 1.00

Two or more useful overlapping tokens
    → 0.85

One useful overlapping token
    → 0.55

No useful overlap
    → 0.15
```

So intent alignment and semantic similarity are two different signals.

------------------------------------------------------------------------

# 8. Job understanding

Each job is converted into structured information where possible.

The system can work with:

``` text
Job title
Required skills
Preferred skills
Experience requirement
Education requirement
Location
Job description
Role/domain information
```

The extraction is rule-based and uses the project's skill taxonomy and
other extraction logic.

It is not a perfect human understanding of every job description.

------------------------------------------------------------------------

# 9. Semantic similarity

Semantic similarity answers:

> **"Do the resume and job description talk about similar things?"**

The configured embedding model is:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

It produces a 384-number vector for text.

Conceptually:

``` text
Resume text
    ↓
[384 numbers]

Job description
    ↓
[384 numbers]
```

The system then compares those vectors using cosine similarity.

------------------------------------------------------------------------

# 10. What does a semantic score of 0.75 mean?

Suppose:

``` text
semantic_similarity = 0.75
```

If the application displays it as a percentage:

``` text
0.75 × 100 = 75%
```

If it displays it on a 10-point scale:

``` text
0.75 × 10 = 7.5 / 10
```

So:

``` text
0.75
=
75%
=
7.5 / 10
```

These are just different ways of displaying the same raw value.

------------------------------------------------------------------------

# 11. What 75% semantic similarity actually means

It means:

> **The mathematical representations of the resume and job are
> relatively similar in semantic space.**

It does **not** mean:

-   75% of the words matched
-   75% of the skills matched
-   75% of the job requirements are satisfied
-   the candidate is 75% qualified
-   the candidate has a 75% chance of being hired

For required skills, look at:

``` text
Required Skill Coverage
```

For overall ranking, look at:

``` text
Final Score
```

For the binary learned model, look at:

``` text
Model Probability
```

Those are different numbers with different meanings.

------------------------------------------------------------------------

# 12. What is an embedding?

An embedding is simply a list of numbers representing text in a way that
allows mathematical comparison.

You do not need to know what every number means.

Think of it like putting a piece of text on a map.

Similar meanings tend to be closer together.

For example:

``` text
"machine learning engineer"
```

and:

``` text
"ML engineer building predictive models"
```

can end up relatively close in semantic space even though the wording is
not identical.

------------------------------------------------------------------------

# 13. What is cosine similarity?

Cosine similarity compares the direction of two vectors.

The mathematical formula is:

``` text
cosine_similarity(A, B)
=
(A · B)
/
(|A| × |B|)
```

Here:

``` text
A = resume embedding
B = job embedding
```

The implementation uses the embedding vectors and calculates this
similarity.

The result is used as a semantic signal.

For normal interpretation in this project:

``` text
closer to 1
    → more semantically similar

closer to 0
    → less semantically similar
```

The ranking code also prevents a negative semantic value from reducing
the score below zero.

------------------------------------------------------------------------

# 14. Is this a vector database?

Not exactly.

The current project has an in-memory `VectorIndex`.

The current flow is:

``` text
Job description
      ↓
Embedding
      ↓
In-memory vector index

Resume
      ↓
Embedding
      ↓
Search the vector index
      ↓
Closest jobs
```

MongoDB is used for application/job data.

The current semantic retrieval layer is an in-memory vector index rather
than a dedicated external vector database such as a hosted vector-search
service.

The architecture can be moved toward a dedicated vector-search system
later.

------------------------------------------------------------------------

# 15. Why semantic similarity is not enough

Suppose a job requires:

``` text
Python
PyTorch
Docker
AWS
Kubernetes
```

and the resume contains:

``` text
Python
Pandas
Scikit-learn
```

The resume and job may still be semantically similar because both
discuss:

``` text
machine learning
Python
engineering
models
data
```

But the candidate is missing several explicit requirements.

Therefore:

> **Semantic similarity finds related content. It does not by itself
> decide eligibility.**

------------------------------------------------------------------------

# 16. Required-skill coverage

Required-skill coverage is much easier to interpret.

The formula is:

``` text
Required Skill Coverage
=
Matched Required Skills
/
Total Required Skills
```

Example:

Job requires:

``` text
Python
SQL
Docker
AWS
```

Candidate has:

``` text
Python
SQL
```

Then:

``` text
2 / 4
=
0.50
=
50%
```

So the candidate has:

> **2 of the 4 extracted required skills.**

That is what the 50% means.

------------------------------------------------------------------------

# 17. Preferred skills

The system can also identify preferred/"nice to have" skills.

Required skills are the main qualification signal.

Preferred skills can add a smaller bonus to the skill score.

The current calculation is:

``` text
preferred bonus
=
0.15 × preferred skill coverage
```

The final skill score is capped at:

``` text
1.0
```

Important:

> **Preferred skills do not replace required skills when eligibility is
> checked.**

------------------------------------------------------------------------

# 18. Eligibility

Eligibility is a separate rule from semantic similarity and final
ranking.

The current required-skill thresholds are:

``` text
Required skill coverage < 30%
    → NOT ELIGIBLE

30% to less than 50%
    → PARTIALLY ELIGIBLE

50% or more
    → ELIGIBLE
```

If a job has no required skills extracted:

``` text
→ ELIGIBLE
```

------------------------------------------------------------------------

# 19. Simple example of "not eligible"

Suppose ABC is applying for:

``` text
Machine Learning Engineer
```

The system extracts these required skills:

``` text
Python
PyTorch
Docker
AWS
Kubernetes
```

ABC's resume shows:

``` text
Python
```

So:

``` text
Matched = 1
Required = 5

1 / 5 = 0.20 = 20%
```

The current rule says:

``` text
20% < 30%
```

Therefore:

> **ABC is not eligible according to the current required-skill coverage
> rule.**

That does **not** mean ABC can never get the job.

It means:

> **The system did not find enough of the extracted required skills in
> the candidate profile to pass the current eligibility threshold.**

------------------------------------------------------------------------

# 20. Partially eligible example

Suppose:

``` text
Required skills = 10
Matched skills = 4
```

Then:

``` text
4 / 10 = 40%
```

So:

``` text
PARTIALLY ELIGIBLE
```

In simple English:

> **The candidate has some of the required skills, but several are still
> missing.**

------------------------------------------------------------------------

# 21. Eligible example

Suppose:

``` text
Required skills = 10
Matched skills = 6
```

Then:

``` text
6 / 10 = 60%
```

So:

``` text
ELIGIBLE
```

This means:

> **The candidate passed the current required-skill coverage rule.**

It does not guarantee an interview or hiring.

------------------------------------------------------------------------

# 22. Experience matching

The system also compares:

``` text
Candidate experience
```

with:

``` text
Job's required experience
```

If the candidate meets the requirement, the match can be:

``` text
1.0
```

If the candidate is below the requirement, the system uses a ratio with
a minimum floor.

For example:

``` text
Candidate = 2 years
Required = 4 years
```

gives approximately:

``` text
2 / 4 = 0.50
```

So experience contributes a lower score.

------------------------------------------------------------------------

# 23. Education matching

Education is represented using a simple internal level.

If:

``` text
Candidate education level >= required level
```

the education match is:

``` text
1.0
```

If the candidate is below the requested level, the score decreases with
a minimum floor.

This is a simple rule-based signal, not a complete evaluation of
academic quality.

------------------------------------------------------------------------

# 24. Location matching

Location matching is intentionally simple.

The current logic includes cases such as:

``` text
No location preference
    → 0.7

Remote job
    → 1.0

Matching location
    → 1.0

No match
    → 0.3
```

This is not a sophisticated geographic-distance model.

------------------------------------------------------------------------

# 25. Location limitation

The current job-location extraction recognizes a fixed set of locations
and remote/hybrid/on-site terms.

When no recognized location is found, the extraction layer can fall back
to:

``` text
Remote
```

That can be incorrect for a real job.

Therefore location matching should currently be treated as a useful but
relatively coarse signal.

------------------------------------------------------------------------

# 26. The deterministic hybrid score

The default ranking method combines six signals:

  Signal                  Default weight
  --------------------- ----------------
  Career intent                     0.20
  Semantic similarity               0.25
  Hard skill match                  0.25
  Experience match                  0.12
  Education match                   0.08
  Location match                    0.10

The weights add up to:

``` text
1.00
```

The formula is:

``` text
Final Score =
    0.20 × Intent
  + 0.25 × Semantic
  + 0.25 × Skill
  + 0.12 × Experience
  + 0.08 × Education
  + 0.10 × Location
```

These weights are configurable.

------------------------------------------------------------------------

# 27. Worked final-score example

Suppose a candidate/job pair has:

``` text
Intent       = 0.85
Semantic     = 0.75
Skill        = 0.75
Experience   = 0.80
Education    = 1.00
Location     = 1.00
```

Calculate:

``` text
0.20 × 0.85 = 0.1700
0.25 × 0.75 = 0.1875
0.25 × 0.75 = 0.1875
0.12 × 0.80 = 0.0960
0.08 × 1.00 = 0.0800
0.10 × 1.00 = 0.1000
```

Add them:

``` text
0.1700
+ 0.1875
+ 0.1875
+ 0.0960
+ 0.0800
+ 0.1000
= 0.821
```

So:

``` text
Final Score = 0.821
```

It can be displayed as:

``` text
82.1%
```

or:

``` text
8.21 / 10
```

if the UI chooses that display format.

Again:

> **82.1% is a ranking score, not an 82.1% hiring probability.**

------------------------------------------------------------------------

# 28. Eligibility and final score are different

This is one of the most important design decisions.

A job can have:

``` text
High semantic similarity
```

but:

``` text
Very low required-skill coverage
```

and therefore be:

``` text
NOT ELIGIBLE
```

Conversely, a job can have:

``` text
Strong required-skill coverage
```

and therefore be:

``` text
ELIGIBLE
```

without necessarily being the highest-ranked job.

So:

``` text
Eligibility
```

answers:

> "Does this candidate pass the current minimum skill-coverage rule?"

while:

``` text
Final score
```

answers:

> "How strongly should this job be ranked compared with the other jobs
> being considered?"

------------------------------------------------------------------------

# 29. Intent gating

Career intent can also limit a recommendation.

The current intent gate uses:

``` text
intent alignment <= 0.20
```

as the low-alignment condition.

When this happens, the final score can be capped at:

``` text
0.35
```

The purpose is:

> **A job that is strongly unrelated to the candidate's stated career
> direction should not become a top recommendation only because the
> resume happens to look similar to it.**

Example:

``` text
Resume:
Data Analyst

Career goal:
GenAI Engineer

Job:
Data Analyst
```

The resume may fit the job very well.

But if the job does not fit the stated career direction, intent gating
can lower its priority.

------------------------------------------------------------------------

# 30. Retrieval and ranking are separate

The recommendation system has two important stages.

## Stage 1 --- Retrieval

The system finds a shortlist of potentially relevant jobs.

The current shortlist target is:

``` text
max(top_k × 5, 30)
```

So if:

``` text
top_k = 10
```

the system initially retrieves up to:

``` text
50
```

jobs.

## Stage 2 --- Ranking

Those jobs are then scored using the ranking signals.

The flow is:

``` text
All available jobs
      ↓
Vector retrieval
      ↓
Shortlist
      ↓
Intent-based shortlist expansion when applicable
      ↓
Detailed scoring
      ↓
Eligibility separation
      ↓
Ranking
      ↓
Top recommendations
```

------------------------------------------------------------------------

# 31. Intent-based shortlist expansion

Vector retrieval is based mainly on the candidate's existing resume
content.

That can create a problem.

Example:

``` text
Resume:
Data Analyst

Career goal:
GenAI Engineer
```

The resume may contain very little GenAI vocabulary.

A purely resume-based vector search could therefore miss useful GenAI
jobs before the ranking layer even sees them.

The current recommender addresses this by adding up to a limited number
of jobs whose title/domain aligns with the normalized intent.

This gives the ranking layer a chance to evaluate them.

------------------------------------------------------------------------

# 32. Important: current "keyword mode" is not a pure keyword search

The project has ranking modes named:

``` text
keyword
vector
hybrid
```

However, the current recommender still uses vector retrieval to create
the initial shortlist even in keyword mode.

The code explicitly documents this.

Therefore the current evaluation should **not** be described as:

> "A completely independent keyword-search engine versus a vector-search
> engine."

A more accurate description is:

> **"Different ranking/scoring modes are compared after a broad
> vector-based shortlist is retrieved."**

A true inverted-index keyword retrieval system would be a future
improvement.

------------------------------------------------------------------------

# 33. Eligible, partially eligible and not eligible jobs

After scoring, the system separates jobs into:

``` text
Eligible
Partially eligible
Not eligible
```

The current recommendation ordering is:

``` text
Eligible jobs
    ↓
sorted by final score

If there are not enough eligible jobs:
    ↓
partially eligible jobs can fill remaining slots
```

Not-eligible jobs are kept separate instead of being treated as normal
recommendations.

------------------------------------------------------------------------

# 34. Recommendation actions

The system can assign actions such as:

``` text
APPLY_NOW
TAILOR_RESUME_FIRST
BUILD_MISSING_EVIDENCE
LOW_PRIORITY
```

These are recommendation actions.

They do not mean:

``` text
guaranteed interview
```

or:

``` text
guaranteed hiring
```

They are intended to tell the candidate what to do next based on the
current ranking and skill-gap information.

------------------------------------------------------------------------

# 35. "Apply Now" does not mean automatic application

The application does not claim to submit the candidate's application
automatically.

The recommendation can provide an external job/application URL when
available.

The user remains responsible for continuing the actual application.

So the correct interpretation is:

``` text
Recommendation
    ↓
Apply Now / external application link
    ↓
Candidate continues the application
```

------------------------------------------------------------------------

# 36. Explanation layer

The ranking calculation and explanation generation are separated.

The flow is:

``` text
Ranking
   ↓
Score + eligibility + skill gaps + intent
   ↓
Retrieve supporting evidence
   ↓
Generate explanation
   ↓
Return explanation
```

The explanation layer does not decide the ranking score.

------------------------------------------------------------------------

# 37. What "RAG" means in this project

The explanation layer retrieves evidence from:

``` text
Resume
Job description
Skill taxonomy
```

The evidence is focused on information relevant to the recommendation,
especially matched and missing skills.

The explanation is then generated from that evidence.

The system also has a groundedness check: generated reasons must be tied
to retrieved evidence.

------------------------------------------------------------------------

# 38. Important clarification about the LLM

The default explanation provider is:

``` text
LocalTemplateLLMProvider
```

Despite the provider name, this default implementation is
**deterministic template-based generation**.

It does not require an external LLM API key.

There is an Anthropic provider interface/stub for a future real
generative provider.

Therefore this README does **not** claim:

> "The default application calls a powerful external LLM for every
> explanation."

The accurate statement is:

> **The project has a grounded explanation layer with a deterministic
> local default and a provider interface for a future/optional
> generative LLM.**

------------------------------------------------------------------------

# 39. Why the explanation layer is separated from ranking

This is useful because the ranking remains reproducible.

If the explanation provider changes, the underlying deterministic
ranking calculation does not need to change.

The system therefore has:

``` text
Ranking
=
calculation

Explanation
=
communication of the calculation
```

That is an important architectural separation.

------------------------------------------------------------------------

# 40. Vector retrieval implementation

The current vector index is an in-memory index.

When jobs are indexed:

``` text
Job descriptions
    ↓
Embedding provider
    ↓
Job embeddings
    ↓
VectorIndex
```

When a candidate is searched:

``` text
Resume
    ↓
Resume embedding
    ↓
VectorIndex search
    ↓
Nearest jobs
```

The current implementation performs a direct similarity search over the
stored vectors.

A dedicated production vector database/search engine can be added later
if the job corpus becomes large enough to require it.

------------------------------------------------------------------------

# 41. Embedding provider

The current configuration uses:

``` text
sentence_transformers
```

with:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

The provider interface also supports alternative implementations.

The important point for the current project is:

> **The normal configured semantic search path uses the local Sentence
> Transformer model rather than a paid embedding API.**

------------------------------------------------------------------------

# 42. MongoDB and Redis

## MongoDB

MongoDB is used for application data such as jobs, candidates,
recommendations and related records.

## Redis

Redis can be used for recommendation caching.

The cache key includes recommendation-related inputs such as:

-   candidate
-   ranking mode
-   top-k
-   filters
-   intent information
-   posting-time filter
-   discovery location

A cache hit can return previously generated recommendations without
repeating the full ranking process.

------------------------------------------------------------------------

# 43. Latency

Latency means:

> **How long the measured recommendation operation takes.**

The application measures elapsed time using a high-resolution timer.

It records:

``` text
latency_ms
```

in milliseconds.

Example:

``` text
100 ms
=
0.1 seconds
```

because:

``` text
1000 ms = 1 second
```

------------------------------------------------------------------------

# 44. Average latency

If five requests take:

``` text
100 ms
200 ms
150 ms
250 ms
300 ms
```

then:

``` text
100 + 200 + 150 + 250 + 300
=
1000 ms
```

and:

``` text
1000 / 5
=
200 ms
```

So average latency is:

``` text
200 ms
```

This means:

> **The measured requests took 200 ms on average.**

It does not mean every request takes exactly 200 ms.

------------------------------------------------------------------------

# 45. Cache and latency

There are two different situations:

``` text
Cache hit
    ↓
Existing recommendation returned quickly
```

and:

``` text
Cache miss
    ↓
Retrieval
    ↓
Scoring
    ↓
Explanation
    ↓
Saving/caching
```

The cache-hit path can therefore be much faster.

For serious production performance testing, it would also be useful to
report:

``` text
p50
p95
p99
```

in addition to average latency.

------------------------------------------------------------------------

# 46. Optional learned ranking model

The repository contains a learned ranking experiment.

The production-style learned ranker uses:

``` text
Candidate/job features
      ↓
StandardScaler
      ↓
LogisticRegression
      ↓
Predicted relevance probability
```

The model uses 15 features in the current `CandidateJobFeatures`
structure.

These include:

1.  semantic similarity
2.  intent alignment
3.  required-skill coverage
4.  preferred-skill coverage
5.  matched required-skill count
6.  missing required-skill count
7.  experience match
8.  education match
9.  location match
10. seniority match
11. domain match
12. resume experience years
13. required experience years
14. skill-gap ratio
15. intent confidence

------------------------------------------------------------------------

# 47. Important: the learned ranker is NOT the default

The recommendation function currently defaults to:

``` text
use_learned_ranker = False
```

So the normal ranking path is:

``` text
Deterministic hybrid baseline
```

not:

``` text
Learned Logistic Regression
```

The learned model is currently an optional/experimental path.

------------------------------------------------------------------------

# 48. What the learned model predicts

The learned model is trained for a binary relevance label:

``` text
1 = positive/relevant training example
0 = negative training example
```

It produces a probability-like score for the positive class.

For example:

``` text
0.82
```

means:

> **The trained classifier gives this candidate-job pair a high
> positive-class probability.**

It does not mean:

> **82% chance of getting hired.**

------------------------------------------------------------------------

# 49. Training examples

The offline dataset-building pipeline creates candidate-job pairs.

A typical positive example is:

``` text
Candidate resume
+
target job role
+
label = 1
```

Negative examples are constructed from other job roles.

Some negative examples are deliberately made harder by selecting jobs
that are somewhat related.

------------------------------------------------------------------------

# 50. What a negative training label means

A dataset label of:

``` text
0
```

does not necessarily mean:

> "A real employer rejected this candidate."

It means:

> **This candidate-job pair was constructed/labeled as a negative
> example for training.**

That distinction is important.

------------------------------------------------------------------------

# 51. Candidate-level train/test split

The dataset builder intentionally splits by candidate rather than
randomly splitting individual rows.

For example:

``` text
Candidate A
    → train

Candidate B
    → train

Candidate C
    → test
```

rather than:

``` text
Candidate A + Job 1
    → train

Candidate A + Job 2
    → test
```

This helps prevent the same candidate's information from appearing on
both sides.

That is a good protection against data leakage.

------------------------------------------------------------------------

# 52. StandardScaler

Different features naturally have different numeric ranges.

For example:

``` text
semantic similarity = 0.75
```

while:

``` text
experience years = 5
```

and:

``` text
matched skill count = 8
```

StandardScaler puts the features onto a more comparable scale before
Logistic Regression learns from them.

------------------------------------------------------------------------

# 53. Balanced Logistic Regression

The training scripts use:

``` text
class_weight="balanced"
```

This gives extra consideration to the smaller class when positive and
negative examples are unevenly distributed.

It does not magically make the dataset balanced.

It changes how the classifier treats the classes during training.

------------------------------------------------------------------------

# 54. Current learned-feature cleanup issue

There is a known redundancy in the current feature design:

``` text
skill_gap_ratio
```

is currently assigned the same value as:

``` text
required_skill_coverage
```

So these two features contain duplicate information.

The model can still run, but this is not an ideal feature design and
should be cleaned up in a future model version.

------------------------------------------------------------------------

# 55. Another learned-feature consistency issue

The hard-skill matcher normalizes skill names before matching.

The learned feature for preferred-skill coverage uses a more direct set
comparison.

That means formatting differences such as:

``` text
Python
```

vs:

``` text
python
```

can be handled differently by different parts of the system.

A future cleanup should use the same normalization everywhere.

------------------------------------------------------------------------

# 56. Learned-ranker integration status

The learned-ranker code exists and can be requested explicitly.

However, the current recommender's ML branch should **not** be described
as the stable/default production ranking system.

The current default remains the deterministic hybrid baseline.

The learned path should be end-to-end tested again before being treated
as the main production ranking path.

------------------------------------------------------------------------

# 57. Offline evaluation

The project evaluates ranking using:

``` text
Precision@K
Recall@K
NDCG@K
```

The evaluation uses hand-labeled relevance information.

These metrics answer:

> **"Did the system put relevant jobs near the top?"**

They do not answer:

> "Did the candidate get hired?"

------------------------------------------------------------------------

# 58. Precision@K

`K` means the number of top results we are examining.

For example:

``` text
Precision@3
```

means:

> "Of the first 3 jobs, how many are relevant?"

Formula:

``` text
Precision@K
=
Relevant jobs in top K
/
K
```

Example:

``` text
Top 3:
Relevant
Relevant
Not relevant
```

Then:

``` text
Precision@3
=
2 / 3
=
66.7%
```

------------------------------------------------------------------------

# 59. Recall@K

Recall@K asks:

> **"Of all relevant jobs available in the evaluation case, how many
> appeared in the first K?"**

Example:

``` text
5 relevant jobs exist
3 appear in top 3
```

Then:

``` text
Recall@3
=
3 / 5
=
60%
```

------------------------------------------------------------------------

# 60. NDCG@K

NDCG means:

``` text
Normalized Discounted Cumulative Gain
```

The simple meaning is:

> **Relevant jobs near the top are better than relevant jobs buried
> lower down.**

For example:

``` text
Rank 1 → Relevant
Rank 2 → Relevant
Rank 3 → Not relevant
```

is better than:

``` text
Rank 1 → Not relevant
Rank 2 → Relevant
Rank 3 → Relevant
```

even though both lists contain two relevant jobs.

NDCG captures that difference.

The evaluation implementation discounts lower-ranked relevant results
and normalizes against an ideal ranking.

------------------------------------------------------------------------

# 61. Current reported ranking results

The repository currently reports:

  Method      Precision@3   Recall@3   NDCG@3
  --------- ------------- ---------- --------
  Keyword           0.429      0.738    0.716
  Vector            0.381      0.762    0.814
  Hybrid            0.524      0.929    0.880

The intent comparison is:

  Variant            Precision@3   Recall@3   NDCG@3
  ---------------- ------------- ---------- --------
  Without Intent           0.429      0.738    0.760
  With Intent              0.524      0.929    0.881

These are measurements from the project's current small hand-labeled
evaluation set.

They should be described as:

> **Project-level evaluation results on the current labeled sample.**

They should not be described as universal real-world accuracy.

------------------------------------------------------------------------

# 62. Why the current evaluation numbers are limited

The current hand-labeled evaluation set contains:

``` text
7 candidates × 12 sample jobs
```

That is useful for checking whether the evaluation method works and for
comparing the current approaches.

It is not large enough to make strong claims about all job seekers or
all job markets.

So:

``` text
Recall@3 = 0.929
```

means:

> **Recall@3 was 0.929 on this evaluation set.**

It does not mean:

> **The application finds 92.9% of all good jobs in the real world.**

------------------------------------------------------------------------

# 63. Why accuracy alone is not enough

For a recommender, the most important question is often:

> **"Are the useful jobs near the top?"**

A model could classify many examples correctly while still putting the
best jobs too low in the list.

That is why ranking metrics are important here.

------------------------------------------------------------------------

# 64. Classification metrics for the learned model

The training scripts also calculate:

``` text
Accuracy
Precision
Recall
F1
ROC AUC
Confusion Matrix
```

These are useful for examining the learned binary classifier.

They are not the same thing as the final recommendation ranking metrics.

------------------------------------------------------------------------

# 65. Accuracy

Accuracy asks:

> **"Out of all examples, how many predictions were correct?"**

Formula:

``` text
Accuracy
=
Correct predictions
/
All predictions
```

Example:

``` text
90 correct
out of 100
=
90%
```

------------------------------------------------------------------------

# 66. Precision

Precision asks:

> **"When the model says an example is relevant, how often is it
> correct?"**

Formula:

``` text
Precision
=
TP / (TP + FP)
```

Example:

``` text
10 predicted relevant
8 actually relevant
```

Then:

``` text
Precision = 8 / 10 = 80%
```

------------------------------------------------------------------------

# 67. Recall

Recall asks:

> **"Of all the relevant examples, how many did the model find?"**

Formula:

``` text
Recall
=
TP / (TP + FN)
```

Example:

``` text
20 relevant examples exist
15 found
```

Then:

``` text
Recall = 15 / 20 = 75%
```

------------------------------------------------------------------------

# 68. F1

F1 combines precision and recall.

Formula:

``` text
F1
=
2 × Precision × Recall
/
(Precision + Recall)
```

Example:

``` text
Precision = 0.80
Recall = 0.60
```

Then:

``` text
F1 ≈ 0.686
```

or:

``` text
68.6%
```

F1 is useful when we want a balance between:

``` text
not recommending too many bad examples
```

and:

``` text
not missing too many good examples
```

------------------------------------------------------------------------

# 69. Confusion matrix

The four basic outcomes are:

  Actual         Model says     Name
  -------------- -------------- ----------------
  Relevant       Relevant       True Positive
  Not relevant   Relevant       False Positive
  Relevant       Not relevant   False Negative
  Not relevant   Not relevant   True Negative

In a job recommender:

### False Positive

The model thinks a job is relevant, but the evaluation label says it is
not.

### False Negative

The model misses a job that the evaluation label says is relevant.

------------------------------------------------------------------------

# 70. ROC

ROC stands for:

``` text
Receiver Operating Characteristic
```

The simple meaning is:

> **It checks how well the classifier separates positive and negative
> examples as the decision threshold changes.**

The classifier can produce values such as:

``` text
0.91
0.74
0.62
0.41
0.18
```

Changing the threshold changes which examples are called positive.

ROC examines that behavior across thresholds.

------------------------------------------------------------------------

# 71. AUC / ROC-AUC

AUC means:

``` text
Area Under the ROC Curve
```

A simple interpretation is:

> **How well does the model separate positive examples from negative
> examples across thresholds?**

Roughly:

``` text
0.50
→ around random

Closer to 1.00
→ stronger separation on the evaluated data
```

AUC does not mean:

``` text
chance of getting hired
```

------------------------------------------------------------------------

# 72. Hit@1

Hit@1 asks:

> **"Was a positive result ranked first?"**

Example:

``` text
10 candidates
7 have a positive result at rank 1
```

Then:

``` text
Hit@1 = 7 / 10 = 70%
```

------------------------------------------------------------------------

# 73. Hit@3

Hit@3 asks:

> **"Did at least one positive result appear in the first three
> positions?"**

Example:

``` text
10 candidates
9 have a positive result in their top 3
```

Then:

``` text
Hit@3 = 90%
```

------------------------------------------------------------------------

# 74. MRR

MRR means:

``` text
Mean Reciprocal Rank
```

It rewards a positive result being near the top.

If the first positive is at:

``` text
rank 1 → 1/1 = 1.00
rank 2 → 1/2 = 0.50
rank 3 → 1/3 ≈ 0.333
rank 5 → 1/5 = 0.20
```

The values are averaged across candidates.

Higher MRR means the first positive result tends to appear higher.

------------------------------------------------------------------------

# 75. Latency vs. quality

These are different measurements.

``` text
Quality
=
How good are the recommendations?

Latency
=
How quickly are they produced?
```

A model can be:

``` text
fast but poor
```

or:

``` text
slow but accurate
```

A useful application needs an acceptable balance of both.

------------------------------------------------------------------------

# 76. Current training-data pipeline

The project also contains an offline dataset-building pipeline.

The pipeline:

``` text
Raw resume/job-role data
        ↓
Candidate objects
        ↓
Job objects
        ↓
Positive candidate-job pairs
        ↓
Negative candidate-job pairs
        ↓
15 candidate/job features
        ↓
Candidate-level train/test split
        ↓
CSV training data
        ↓
Optional model training
```

The dataset-building script itself does **not** train the model.

It prepares the data.

------------------------------------------------------------------------

# 77. Important distinction: raw resumes vs. training examples

If a raw dataset contains:

``` text
10,000 resumes
```

that does not automatically mean:

``` text
10,000 model training examples
```

The training pipeline can create multiple candidate-job pairs from one
candidate.

Therefore the final number of training rows can be much larger than the
number of original resumes.

------------------------------------------------------------------------

# 78. Real user feedback pipeline

There is also a separate learning pipeline designed around
recommendation feedback.

The idea is:

``` text
Recommendation
      ↓
User action/feedback
      ↓
Learning example
      ↓
Future model training
```

This is different from the offline role-labeled dataset.

Real feedback can eventually provide better evidence about what
candidates actually find useful.

However:

> **User rejection is not automatically the same thing as "bad
> recommendation."**

A person might reject a good job because of salary, timing, location,
company preference, or another reason that the ranking model does not
currently represent.

------------------------------------------------------------------------

# 79. Current default vs. experimental ML

The most accurate way to describe the project today is:

``` text
DEFAULT:
Deterministic hybrid ranking

OPTIONAL / EXPERIMENTAL:
Learned Logistic Regression re-ranking
```

This distinction should remain visible in the documentation.

------------------------------------------------------------------------

# 80. What is currently working conceptually

The current implementation provides these core pieces:

-   resume extraction
-   career-intent normalization
-   job requirement extraction
-   semantic embeddings
-   in-memory vector retrieval
-   intent-aware shortlist expansion
-   deterministic hybrid scoring
-   hard-skill eligibility
-   skill-gap reporting
-   recommendation actions
-   grounded explanation generation
-   recommendation caching
-   ranking evaluation
-   an optional learned-ranker/training pipeline

------------------------------------------------------------------------

# 81. What should NOT be described as already production-complete

The README should not claim that the project already has:

-   a dedicated external vector database
-   a production-grade learned ranking model as the default
-   real-world hiring prediction
-   a large statistically representative benchmark
-   a live universal job market feed
-   automatic application submission
-   a production authentication/security/data-retention system
-   a fully generative LLM as the default explanation engine

Those are either future directions, optional interfaces, experimental
components, or outside the scope of the current implementation.

------------------------------------------------------------------------

# 82. Current limitations

## 1. Small evaluation benchmark

The current hand-labeled benchmark is small.

## 2. Location extraction is coarse

Unrecognized locations can be interpreted incorrectly.

## 3. Intent gating can be strict

Related jobs with different titles can receive weaker intent alignment.

## 4. Keyword mode is not a pure keyword retrieval pipeline

The shortlist is still vector-retrieved.

## 5. Learned ranker is not the default

The deterministic hybrid ranker remains the main ranking path.

## 6. Learned feature redundancy

`skill_gap_ratio` currently duplicates `required_skill_coverage`.

## 7. Training labels are not hiring outcomes

Offline positive/negative labels are not employer hiring decisions.

## 8. Application links depend on available job data

The system can resolve/use an external URL when the job data and
resolver provide one. It does not automatically submit applications.

------------------------------------------------------------------------

# 83. Technology stack

## Frontend

-   Next.js
-   React
-   TypeScript

## Backend

-   Python
-   FastAPI

## Data / infrastructure

-   MongoDB
-   Redis
-   Docker

## AI / ranking

-   Sentence Transformers
-   embeddings
-   in-memory vector index
-   deterministic hybrid ranking
-   skill taxonomy
-   intent normalization
-   optional Logistic Regression ranker
-   grounded explanation layer

## Evaluation

-   Precision@K
-   Recall@K
-   NDCG@K
-   Accuracy
-   Precision
-   Recall
-   F1
-   ROC-AUC
-   Hit@1
-   Hit@3
-   MRR
-   latency measurement

------------------------------------------------------------------------

# 84. Why the project uses both semantic and explicit matching

Semantic matching is useful for:

``` text
meaning
context
related concepts
```

Explicit skill matching is useful for:

``` text
required tools
required technologies
clear skill gaps
```

For this application, the two signals answer different questions.

The intended design is therefore:

``` text
Semantic similarity
        +
Explicit requirements
        +
Career intent
        +
Other fit signals
        ↓
Better ranking
```

------------------------------------------------------------------------

# 85. Example: the full decision

Suppose:

``` text
Semantic similarity = 0.75
Required skill coverage = 0.60
Experience match = 1.00
Education match = 1.00
Location match = 1.00
Intent alignment = 0.85
```

The candidate:

``` text
passes eligibility
```

because:

``` text
0.60 >= 0.50
```

The final hybrid score is:

``` text
0.20(0.85)
+
0.25(0.75)
+
0.25(0.60)
+
0.12(1.00)
+
0.08(1.00)
+
0.10(1.00)
```

which equals:

``` text
0.7825
```

So the job has:

``` text
Final ranking score = 78.25%
```

while:

``` text
Required skill coverage = 60%
```

These numbers mean different things.

------------------------------------------------------------------------

# 86. How a user should read a recommendation

A beginner-friendly recommendation should be understood in this order:

### 1. Eligibility

``` text
Eligible
Partially eligible
Not eligible
```

### 2. Required skills

Look at:

``` text
Matched
Missing
```

### 3. Career direction

Ask:

> Does this job match what I said I want to do?

### 4. Overall ranking

Use the final score to compare this job with other jobs.

### 5. Explanation

Read why the job was recommended.

### 6. Action

Decide whether to:

``` text
Apply
Tailor resume
Build missing evidence
Lower priority
```

------------------------------------------------------------------------

# 87. The most important distinction in the project

Remember:

``` text
Semantic similarity
        ≠
Skill coverage
        ≠
Eligibility
        ≠
Final ranking score
        ≠
ML probability
        ≠
Hiring probability
```

They answer different questions.

------------------------------------------------------------------------

# 88. One-line meaning of every important number

  -----------------------------------------------------------------------
  Number / Metric                     Meaning
  ----------------------------------- -----------------------------------
  Semantic 0.75                       Resume and job have fairly similar
                                      semantic representations

  75% semantic                        Same 0.75 value shown as a
                                      percentage

  7.5/10 semantic                     Same 0.75 value shown on a 10-point
                                      display

  Skill coverage 0.75                 75% of extracted required skills
                                      were matched

  Final score 0.75                    Weighted ranking score of
                                      approximately 0.75

  ML probability 0.75                 Learned classifier's positive-class
                                      probability

  Eligible                            Passed the current required-skill
                                      coverage rule

  Precision                           How often positive predictions are
                                      correct

  Recall                              How many real positives were found

  F1                                  Balance of precision and recall

  ROC-AUC                             Positive/negative separation
                                      quality across thresholds

  P@3                                 How many of the first 3 results are
                                      relevant

  R@3                                 How many relevant results were
                                      found in the first 3

  NDCG@3                              How well relevant results are
                                      placed near the top

  Hit@1                               Positive result is first

  Hit@3                               At least one positive is in the
                                      first 3

  MRR                                 How high the first positive appears
                                      on average

  Latency                             Time taken by the measured
                                      operation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 89. API overview

The backend exposes versioned API routes under:

``` text
/api/v1
```

The repository currently documents endpoints for areas including:

  Method   Endpoint                            Purpose
  -------- ----------------------------------- ----------------------------------
  POST     `/candidates/resume`                Upload/process a resume
  GET      `/candidates/{id}/resume`           Fetch parsed resume
  POST     `/intent`                           Save career intent
  GET      `/intent/preview?text=`             Preview intent normalization
  GET      `/intent/{candidate_id}`            Get current intent
  POST     `/jobs`                             Add a job
  POST     `/jobs/bulk`                        Add multiple jobs
  POST     `/jobs/reindex`                     Rebuild job index
  POST     `/recommendations/rank`             Rank jobs
  POST     `/recommendations/compare`          Compare ranking approaches
  GET      `/recommendations/{candidate_id}`   Get recommendations
  POST     `/feedback`                         Record recommendation feedback
  GET      `/evaluation/run`                   Run evaluation
  GET      `/evaluation/intent-impact`         Compare intent/no-intent ranking

Interactive API documentation is available when the backend is running:

``` text
http://localhost:8000/docs
```

------------------------------------------------------------------------

# 90. Project structure

The main project areas are:

``` text
backend/
    app/
        api/
        models/
        repositories/
        services/
            embeddings.py
            extraction.py
            intent.py
            ranking.py
            recommender.py
            rag.py
            learned_ranker.py
            evaluation.py
            ...
    data/
    scripts/
    tests/

frontend/
    components/
    app/
    ...

evaluation/
    run_eval.py
    ...

postman/
    ...

docker-compose.yml
README.md
```

------------------------------------------------------------------------

# 91. Demo mode

The repository is configured for a demo-oriented local setup.

The current configuration includes:

``` text
DEMO_MODE = True
SEED_DEMO_DATA = True
```

The project can therefore be run locally without requiring a production
deployment architecture.

------------------------------------------------------------------------

# 92. Quick start

## Backend

``` bash
cd backend

python -m venv .venv
```

### Windows

``` bash
.venv\Scripts\activate
```

### macOS / Linux

``` bash
source .venv/bin/activate
```

Then:

``` bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

``` text
http://localhost:8000/docs
```

------------------------------------------------------------------------

# 93. Frontend

In another terminal:

``` bash
cd frontend
npm install
npm run dev
```

Then open:

``` text
http://localhost:3000
```

The intended basic flow is:

``` text
Dashboard
   ↓
Upload/use resume
   ↓
Enter career goal
   ↓
View recommendations
```

------------------------------------------------------------------------

# 94. Tests

The repository contains tests for important areas including:

-   resume extraction
-   intent normalization
-   intent gating
-   ranking
-   evaluation metrics
-   API behavior

Run:

``` bash
cd backend
pytest -v
```

The exact test results depend on the current checkout and environment.

------------------------------------------------------------------------

# 95. Evaluation

The repository contains an evaluation harness.

A typical evaluation command is:

``` bash
python evaluation/run_eval.py --base-url http://localhost:8000
```

The backend must be running for API-based evaluation.

The intent-impact endpoint is also available:

``` text
GET /api/v1/evaluation/intent-impact
```

The evaluation should be interpreted as a benchmark of the current
labeled cases, not as proof of real-world hiring success.

------------------------------------------------------------------------

# 96. Production direction

The current project should be understood as a working
application/prototype architecture with a clear path toward larger-scale
deployment.

Possible future replacements include:

``` text
Current in-memory vector index
        ↓
Dedicated vector search

Local/demo infrastructure
        ↓
Production managed infrastructure

Local template explanation provider
        ↓
Production generative provider if needed

Small hand-labeled evaluation set
        ↓
Large real-world benchmark
```

These are future deployment improvements, not claims about what the
current system already uses.

------------------------------------------------------------------------

# 97. What should be improved next?

The highest-value improvements are:

## 1. Larger evaluation benchmark

More candidates, more jobs and more independent relevance labels.

## 2. Better location extraction

Avoid treating unknown locations as automatically remote.

## 3. True keyword retrieval baseline

If keyword-vs-vector comparisons are important, implement a genuine
keyword/inverted-index retrieval path.

## 4. Clean up learned features

Remove or redefine duplicated features.

## 5. Consistent skill normalization

Use the same normalization logic everywhere.

## 6. Validate the learned-ranker path

Run a complete end-to-end test before making it the default.

## 7. Better latency reporting

Add:

``` text
p50
p95
p99
```

rather than relying only on averages.

## 8. Larger real-user feedback dataset

Real interactions can eventually provide better ranking labels.

------------------------------------------------------------------------

# 98. What the project can honestly say today

A technically accurate description is:

> **Job Application Strategy AI is an intent-aware job recommendation
> system that combines semantic retrieval, explicit requirement
> matching, deterministic hybrid ranking, eligibility rules, skill-gap
> analysis, and grounded explanations. It also contains an optional
> learned Logistic Regression ranking pipeline for experimentation.**

That description matches the current architecture without pretending
that experimental components are already production defaults.

------------------------------------------------------------------------

# 99. What the project should NOT say

Avoid claims such as:

> "The system predicts whether a company will hire you."

Avoid:

> "75% semantic similarity means you are 75% qualified."

Avoid:

> "AUC 0.90 means a 90% hiring chance."

Avoid:

> "The system uses a production vector database."

Avoid:

> "The system uses an LLM for every explanation."

Avoid:

> "The learned ML ranker is the production ranking engine."

Avoid:

> "92.9% Recall@3 means 92.9% real-world job recommendation accuracy."

These statements would overstate the current implementation.

------------------------------------------------------------------------

# 100. Final mental model

If you remember only one flow, remember this:

``` text
                    RESUME
                       |
                       v
               Candidate Profile
                       |
          +------------+------------+
          |                         |
          v                         v
   Resume Embedding            Career Goal
          |                         |
          v                         v
   Vector Retrieval         Intent Normalization
          |                         |
          +------------+------------+
                       |
                       v
                  Job Shortlist
                       |
                       v
             Detailed Job Matching
                       |
       +---------------+---------------+
       |               |               |
       v               v               v
   Semantic         Skills         Experience
       |               |               |
       +---------------+---------------+
                       |
                 Education/Location
                       |
                       v
                 Intent Alignment
                       |
                       v
               Eligibility Rule
                       |
                       v
                Final Ranking
                       |
                       v
              Skill Gaps + Explanation
                       |
                       v
                    ACTION
```

------------------------------------------------------------------------

# 101. The one sentence to remember

> **Semantic similarity tells us whether the resume and job are talking
> about similar things. Required-skill coverage tells us how many
> extracted requirements are actually present. Eligibility is a separate
> rule. The final score combines several signals to rank jobs.
> Evaluation metrics tell us how the system performs on the available
> test data, while latency tells us how quickly it runs.**

------------------------------------------------------------------------

# 102. Current-status summary

  -----------------------------------------------------------------------
  Area                                Current status
  ----------------------------------- -----------------------------------
  Resume extraction                   Implemented

  Career intent normalization         Implemented

  Semantic embeddings                 Implemented

  In-memory vector retrieval          Implemented

  Dedicated external vector DB        Not currently the default

  Required-skill matching             Implemented

  Eligibility thresholds              Implemented

  Experience matching                 Implemented

  Education matching                  Implemented

  Location matching                   Implemented, but coarse

  Intent gating                       Implemented

  Deterministic hybrid ranking        **Default ranking path**

  Learned Logistic Regression ranker  Available, optional/experimental

  Grounded explanation layer          Implemented

  Default external LLM generation     **No**

  Recommendation caching              Implemented

  Ranking evaluation                  Implemented

  Large statistical benchmark         **No**

  Hiring-outcome prediction           **No**

  Automatic application submission    **No**
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 103. Bottom line

The project is not just:

``` text
Resume → similarity score
```

It is:

``` text
Resume
+
Career goal
+
Job requirements
+
Semantic similarity
+
Skill coverage
+
Experience
+
Education
+
Location
        ↓
Eligibility
        ↓
Hybrid ranking
        ↓
Skill gaps
        ↓
Grounded explanation
        ↓
Application decision
```

And the most important rule for understanding the system is:

> **Never ask only "What is the percentage?" Ask "Percentage of WHAT?"**

A semantic `75%`, skill coverage `75%`, final score `75%`, recall `75%`,
and ML probability `75%` are all different things.

That is why this README deliberately uses plain English and keeps the
current implementation separate from future ideas.
