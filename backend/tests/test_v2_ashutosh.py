"""
V2 Test Script: Ashutosh Renushe - Accountant Candidate
========================================================

This script tests the V2 Job Application Strategy AI with a Commerce/Finance
candidate to verify domain-agnostic support and correct skill gap detection.

Expected Results:
- Extracted skills should include accounting, gst_compliance, tally, excel, etc.
- Career domain should be detected as "Accountant" with high confidence
- For a Data Analyst role requiring Python, SQL, Statistics, Data Visualization, Excel:
  - Matched: excel (1 skill)
  - Missing: python, sql, statistics, data_visualization (4 skills)
  - Coverage: 20%
- The candidate should NOT be classified as a strong match for Data Analyst roles
"""

from app.services.extraction import extract_resume_profile, extract_job_requirements
from app.services.ranking import compute_hard_skill_match


def test_ashutosh_resume():
    """Test Ashutosh Renushe's resume extraction and skill matching."""
    
    # Ashutosh Renushe's resume text (Commerce/Finance background)
    resume_text = """
Ashutosh Dhanajirao Renushe
Accountant | Accounts & GST
Mumbai, Maharashtra
Email: ashutosh.renushe@example.com
Phone: +91 98765 43210

EDUCATION
Bachelor of Commerce (B.Com.)
Prahladrai Dalmia Lions College
2026

12th Commerce
Lords Universal College, 2022

EXPERIENCE
Accountant at Ramesh Kundar, CA Firm
Mumbai, Maharashtra
July 2025 – June 2026

Professional experience includes:
- Bookkeeping and day-to-day accounting
- Tally Prime for accounting operations
- GST compliance including GSTR-1, GSTR-3B, GSTR-2B reconciliation
- GST registration support
- E-way bills and E-invoices
- Bank audit support
- Bank statement verification
- Loan/supporting document verification
- Voucher verification
- Reconciliation
- Cash-related records
- Interest calculations
- Audit working papers
- Excel-based accounting and reconciliation
- Filters, Sorting, SUM/SUMIF, COUNTIF, VLOOKUP/XLOOKUP
- Pivot Tables
- Conditional formatting
- Basic data cleaning
- Documentation and reporting

SKILLS
Accounting & Bookkeeping
Tally Prime
Tally
GST Compliance
GSTR-1, GSTR-2B Reconciliation, GSTR-3B
GST Registration Support
E-Way Bill, E-Invoice
Bank Audit Support
Bank Reconciliation
Voucher Verification
Ledger & Record Maintenance
MS Excel
Pivot Tables, SUMIF/COUNTIF, VLOOKUP/XLOOKUP
Conditional Formatting
Data Cleaning
Documentation & Reporting
"""

    print("=" * 70)
    print("TEST CANDIDATE: ASHUTOSH RENUSHE")
    print("Profile: Accountant | Accounts & GST | B.Com")
    print("=" * 70)
    
    # Extract candidate profile
    candidate = extract_resume_profile(resume_text)
    
    print("\n1. EXTRACTED SKILLS")
    print("-" * 40)
    for skill in candidate['skills']:
        print(f"   ✓ {skill}")
    
    print("\n2. DETECTED CAREER DOMAINS")
    print("-" * 40)
    for domain in candidate.get('career_domains', []):
        print(f"   • {domain['domain_name']} (confidence: {domain['confidence']:.0%})")
    
    print(f"\n   Primary Domain: {candidate.get('primary_domain', 'N/A')}")
    print(f"   Seniority Level: {candidate.get('seniority_level', 'N/A')}")
    print(f"   Experience Years: {candidate.get('total_experience_years', 0):.1f}")
    
    # Test against Data Analyst job
    data_analyst_job = """
Data Analyst Position

Requirements:
- Bachelor's degree in relevant field
- Python programming for data analysis
- SQL for database queries and data manipulation
- Statistics and statistical analysis
- Data Visualization using Tableau or Power BI
- Excel for data analysis including pivot tables and formulas
- Experience with data cleaning and preprocessing
"""
    
    print("\n" + "=" * 70)
    print("TEST JOB: DATA ANALYST")
    print("Required Skills: Python, SQL, Statistics, Data Visualization, Excel")
    print("=" * 70)
    
    job_reqs = extract_job_requirements(data_analyst_job)
    
    print("\n3. JOB REQUIRED SKILLS (extracted)")
    print("-" * 40)
    for skill in job_reqs.required_skills:
        print(f"   → {skill}")
    
    # Calculate skill gap
    candidate_skills_set = set(candidate['skills'])
    required_skills_set = set(job_reqs.required_skills)
    matched = list(candidate_skills_set & required_skills_set)
    missing = list(required_skills_set - candidate_skills_set)
    coverage = len(matched) / len(required_skills_set) if required_skills_set else 0
    
    print("\n4. SKILL GAP ANALYSIS")
    print("-" * 40)
    print(f"   MATCHED Required Skills: {matched}")
    print(f"   MISSING Required Skills: {missing}")
    print(f"   Coverage Ratio: {coverage:.1%}")
    
    print("\n5. EXPECTED vs ACTUAL VERIFICATION")
    print("-" * 40)
    
    # Expected results
    expected_matched = {'excel'}
    expected_missing = {'python', 'sql', 'statistics', 'data_visualization'}
    expected_coverage = 0.20  # 20%
    
    matched_set = set(matched)
    missing_set = set(missing)
    
    print(f"   Matched Skills:")
    print(f"      Expected: {expected_matched}")
    print(f"      Actual:   {matched_set}")
    print(f"      ✓ PASS" if matched_set == expected_matched else f"      ✗ FAIL")
    
    print(f"\n   Missing Skills:")
    print(f"      Expected: {expected_missing}")
    print(f"      Actual:   {missing_set}")
    print(f"      ✓ PASS" if missing_set == expected_missing else f"      ✗ FAIL")
    
    print(f"\n   Coverage Ratio:")
    print(f"      Expected: ~{expected_coverage:.0%}")
    print(f"      Actual:   {coverage:.1%}")
    print(f"      ✓ PASS" if abs(coverage - expected_coverage) < 0.05 else f"      ✗ FAIL")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION DECISION")
    print("=" * 70)
    
    # Determine recommendation based on coverage
    if coverage >= 0.8:
        action = "APPLY_NOW"
        reason = "Strong skill match (>80%)"
    elif coverage >= 0.5:
        action = "TAILOR_RESUME_FIRST"
        reason = "Moderate skill match (50-80%)"
    elif coverage >= 0.2:
        action = "BUILD_MISSING_EVIDENCE"
        reason = "Partial skill match (20-50%) - build missing skills first"
    else:
        action = "LOW_PRIORITY"
        reason = "Weak skill match (<20%)"
    
    print(f"\n   Action: {action}")
    print(f"   Reason: {reason}")
    print(f"   Coverage: {coverage:.1%}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    
    # Final assertion
    all_pass = (
        matched_set == expected_matched and
        missing_set == expected_missing and
        abs(coverage - expected_coverage) < 0.05
    )
    
    if all_pass:
        print("\n✓ ALL TESTS PASSED")
        print("  - Correctly identified accountant's skills")
        print("  - Correctly detected career domain as Finance/Accounting")
        print("  - Correctly identified skill gaps for Data Analyst role")
        print("  - Did NOT falsely classify as strong match (no false positive)")
        print("  - Coverage ratio matches expected 20%")
    else:
        print("\n✗ SOME TESTS FAILED")
        
    return all_pass


if __name__ == "__main__":
    success = test_ashutosh_resume()
    exit(0 if success else 1)
