"""
Visualize Fuzzy Logic Membership Functions
Demonstrates the fluid boundaries used in SMC detection
"""

import numpy as np
import matplotlib.pyplot as plt
from data_preparation_pipeline import FuzzyMembershipFunctions, FuzzySMCClassifier

def plot_candle_body_membership():
    """Visualize candle body size fuzzy classification"""
    fuzzy = FuzzySMCClassifier()
    
    # Generate x values (body size in ATR units)
    x = np.linspace(0, 5, 500)
    
    # Calculate membership for each term
    doji = [fuzzy.classify_candle_body(val)['doji'] for val in x]
    small = [fuzzy.classify_candle_body(val)['small'] for val in x]
    medium = [fuzzy.classify_candle_body(val)['medium'] for val in x]
    large = [fuzzy.classify_candle_body(val)['large'] for val in x]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, doji, 'r-', label='Doji', linewidth=2)
    plt.plot(x, small, 'g-', label='Small', linewidth=2)
    plt.plot(x, medium, 'b-', label='Medium', linewidth=2)
    plt.plot(x, large, 'm-', label='Large', linewidth=2)
    
    plt.xlabel('Candle Body Size (ATR units)', fontsize=12)
    plt.ylabel('Membership Degree', fontsize=12)
    plt.title('Fuzzy Logic: Candle Body Size Classification', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.1)
    
    # Add annotations
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.text(2.5, 0.52, 'Membership = 0.5 (Equal membership)', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig('fuzzy_candle_body.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fuzzy_candle_body.png")

def plot_displacement_membership():
    """Visualize displacement strength fuzzy classification"""
    fuzzy = FuzzySMCClassifier()
    
    # Generate x values (displacement in ATR units)
    x = np.linspace(0, 8, 500)
    
    # Calculate membership for each term
    weak = [fuzzy.classify_displacement(val)['weak'] for val in x]
    moderate = [fuzzy.classify_displacement(val)['moderate'] for val in x]
    strong = [fuzzy.classify_displacement(val)['strong'] for val in x]
    extreme = [fuzzy.classify_displacement(val)['extreme'] for val in x]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, weak, 'orange', label='Weak', linewidth=2)
    plt.plot(x, moderate, 'yellow', label='Moderate', linewidth=2)
    plt.plot(x, strong, 'green', label='Strong', linewidth=2)
    plt.plot(x, extreme, 'red', label='Extreme', linewidth=2)
    
    plt.xlabel('Displacement (ATR units)', fontsize=12)
    plt.ylabel('Membership Degree', fontsize=12)
    plt.title('Fuzzy Logic: Displacement Strength Classification', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.1)
    
    # Add vertical line at traditional threshold
    plt.axvline(x=1.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(1.55, 0.9, 'Traditional Rigid\nThreshold (1.5 ATR)', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('fuzzy_displacement.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fuzzy_displacement.png")

def plot_gap_size_membership():
    """Visualize FVG gap size fuzzy classification"""
    fuzzy = FuzzySMCClassifier()
    
    # Generate x values (gap size in ATR units)
    x = np.linspace(0, 6, 500)
    
    # Calculate membership for each term
    insignificant = [fuzzy.classify_gap_size(val)['insignificant'] for val in x]
    small = [fuzzy.classify_gap_size(val)['small'] for val in x]
    medium = [fuzzy.classify_gap_size(val)['medium'] for val in x]
    large = [fuzzy.classify_gap_size(val)['large'] for val in x]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, insignificant, 'gray', label='Insignificant', linewidth=2)
    plt.plot(x, small, 'cyan', label='Small', linewidth=2)
    plt.plot(x, medium, 'blue', label='Medium', linewidth=2)
    plt.plot(x, large, 'purple', label='Large', linewidth=2)
    
    plt.xlabel('Fair Value Gap Size (ATR units)', fontsize=12)
    plt.ylabel('Membership Degree', fontsize=12)
    plt.title('Fuzzy Logic: FVG Gap Size Classification', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.1)
    
    # Add vertical line at traditional threshold
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(0.55, 0.9, 'Traditional Rigid\nThreshold (0.5 ATR)', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('fuzzy_gap_size.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fuzzy_gap_size.png")

def plot_comparison_rigid_vs_fuzzy():
    """Compare rigid threshold vs fuzzy logic approach"""
    x = np.linspace(0, 10, 500)
    
    # Rigid threshold (step function)
    rigid = np.where(x >= 5, 1, 0)
    
    # Fuzzy triangular membership
    mf = FuzzyMembershipFunctions()
    fuzzy = [mf.triangular(val, 2.5, 5, 7.5) for val in x]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rigid threshold
    ax1.plot(x, rigid, 'r-', linewidth=3)
    ax1.axvline(x=5, color='red', linestyle='--', alpha=0.5)
    ax1.fill_between(x, 0, rigid, alpha=0.3, color='red')
    ax1.set_xlabel('Parameter Value', fontsize=12)
    ax1.set_ylabel('Acceptance (0 or 1)', fontsize=12)
    ax1.set_title('Traditional Rigid Threshold', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.2)
    ax1.text(5.2, 0.5, 'Sharp cutoff at 5.0\n4.99 → Rejected\n5.01 → Accepted', 
             fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Fuzzy logic
    ax2.plot(x, fuzzy, 'g-', linewidth=3)
    ax2.fill_between(x, 0, fuzzy, alpha=0.3, color='green')
    ax2.set_xlabel('Parameter Value', fontsize=12)
    ax2.set_ylabel('Membership Degree [0, 1]', fontsize=12)
    ax2.set_title('Fuzzy Logic Membership Function', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.2)
    ax2.text(5.2, 0.5, 'Smooth transition\n2.5 → 0% membership\n5.0 → 100% membership\n7.5 → 0% membership', 
             fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('fuzzy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fuzzy_comparison.png")

def plot_mtf_proximity_membership():
    """Visualize multi-timeframe proximity fuzzy membership"""
    mf = FuzzyMembershipFunctions()
    
    # Generate x values (distance from HTF structure in ATR units)
    x = np.linspace(0, 5, 500)
    
    # Calculate proximity membership (triangular: peak at 0, zero at 3 ATR)
    proximity = [mf.triangular(val, 0.0, 0.0, 3.0) for val in x]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, proximity, 'b-', linewidth=3, label='HTF Structure Proximity')
    plt.fill_between(x, 0, proximity, alpha=0.3, color='blue')
    
    plt.xlabel('Distance from HTF Structure (ATR units)', fontsize=12)
    plt.ylabel('Confluence Membership Degree', fontsize=12)
    plt.title('Fuzzy Logic: Multi-Timeframe Confluence Proximity', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.1)
    
    # Add annotations
    plt.axvline(x=0, color='green', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(0.05, 0.95, 'Perfect Alignment\n(100% confluence)', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.axvline(x=1.5, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(1.55, 0.5, 'Moderate Proximity\n(~50% confluence)', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.axvline(x=3.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(3.05, 0.1, 'No Influence\n(0% confluence)', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('fuzzy_mtf_proximity.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fuzzy_mtf_proximity.png")

def demonstrate_fuzzy_inference():
    """Demonstrate fuzzy inference for OB quality scoring"""
    fuzzy = FuzzySMCClassifier()
    
    print("\n" + "="*70)
    print("FUZZY LOGIC INFERENCE DEMONSTRATION")
    print("="*70)
    
    # Test cases
    test_cases = [
        (0.8, 0.2, "Weak displacement, small body"),
        (1.5, 0.5, "Moderate displacement, small body"),
        (2.5, 1.2, "Strong displacement, medium body"),
        (4.0, 2.0, "Strong displacement, medium body"),
        (6.0, 3.5, "Extreme displacement, large body")
    ]
    
    print("\nOrder Block Quality Assessment:")
    print("-" * 70)
    
    for disp_atr, body_atr, description in test_cases:
        # Classify displacement
        disp_fuzzy = fuzzy.classify_displacement(disp_atr)
        disp_score = disp_fuzzy['moderate'] + disp_fuzzy['strong'] + disp_fuzzy['extreme']
        
        # Classify body
        body_fuzzy = fuzzy.classify_candle_body(body_atr)
        body_score = body_fuzzy['small'] + body_fuzzy['medium'] + body_fuzzy['large']
        
        # Fuzzy AND for quality
        quality = fuzzy.fuzzy_and(disp_score, body_score)
        
        # Decision
        decision = "✓ VALID" if quality >= 0.3 else "✗ REJECTED"
        
        print(f"\nCase: {description}")
        print(f"  Displacement: {disp_atr:.1f} ATR → Score: {disp_score:.3f}")
        print(f"  Body Size: {body_atr:.1f} ATR → Score: {body_score:.3f}")
        print(f"  Quality (Fuzzy AND): {quality:.3f}")
        print(f"  Decision: {decision}")
    
    print("\n" + "="*70)

def demonstrate_mtf_confluence():
    """Demonstrate multi-timeframe confluence fuzzy logic"""
    fuzzy = FuzzySMCClassifier()
    mf = FuzzyMembershipFunctions()
    
    print("\n" + "="*70)
    print("MULTI-TIMEFRAME CONFLUENCE DEMONSTRATION")
    print("="*70)
    
    # Test cases: (distance_to_htf_ob, htf_ob_quality, description)
    test_cases = [
        (0.0, 0.8, "Price at HTF OB center, high quality"),
        (0.5, 0.7, "Price near HTF OB, good quality"),
        (1.5, 0.6, "Price moderately close to HTF OB"),
        (2.5, 0.5, "Price far from HTF OB"),
        (3.5, 0.4, "Price beyond HTF OB influence")
    ]
    
    print("\nHTF Order Block Confluence Assessment:")
    print("-" * 70)
    
    for distance_atr, htf_quality, description in test_cases:
        # Calculate proximity fuzzy score
        proximity_score = mf.triangular(distance_atr, 0.0, 0.0, 3.0)
        
        # Combine with HTF OB quality using fuzzy AND
        confluence_score = fuzzy.fuzzy_and(proximity_score, htf_quality)
        
        # Decision
        decision = "✓ STRONG CONFLUENCE" if confluence_score >= 0.5 else \
                   "~ MODERATE CONFLUENCE" if confluence_score >= 0.3 else \
                   "✗ WEAK CONFLUENCE"
        
        print(f"\nCase: {description}")
        print(f"  Distance to HTF OB: {distance_atr:.1f} ATR")
        print(f"  Proximity Score: {proximity_score:.3f}")
        print(f"  HTF OB Quality: {htf_quality:.3f}")
        print(f"  Confluence Score (Fuzzy AND): {confluence_score:.3f}")
        print(f"  Assessment: {decision}")
    
    print("\n" + "="*70)
    print("\nKey Insight:")
    print("Fuzzy logic allows gradual decay of HTF influence with distance,")
    print("avoiding rigid 'inside/outside' binary decisions. This mirrors")
    print("institutional behavior where proximity matters, not just overlap.")
    print("="*70)

if __name__ == "__main__":
    print("\n🧠 Generating Fuzzy Logic Visualizations...")
    print("="*70)
    
    # Generate all plots
    plot_candle_body_membership()
    plot_displacement_membership()
    plot_gap_size_membership()
    plot_comparison_rigid_vs_fuzzy()
    plot_mtf_proximity_membership()
    
    # Demonstrate fuzzy inference
    demonstrate_fuzzy_inference()
    demonstrate_mtf_confluence()
    
    print("\n✅ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - fuzzy_candle_body.png")
    print("  - fuzzy_displacement.png")
    print("  - fuzzy_gap_size.png")
    print("  - fuzzy_comparison.png")
    print("  - fuzzy_mtf_proximity.png")
