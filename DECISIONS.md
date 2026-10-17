# Architectural Decisions & Post-Hoc Analysis

### Q1: What was the most technically challenging part of your implementation, and how did you approach it?

The hardest part was choosing a detection method I could trust, because different metrics disagreed. Ranked by near-OOD AUROC on the unseen animal classes, `Entropy` looked best (0.859). When I added SVHN digits as far-OOD, `Entropy` rejected only 47% of them at the validation threshold, while `Energy` rejected 86%. `Energy` also gave the best overall system accuracy (70.3% vs. 68.1% for `Entropy`). I had expected one method to win everywhere, but none of them did. 

To address this, I stopped relying on a single number. I set every threshold on a validation set (the 5th percentile, which keeps about 95% of known images), then evaluated on near-OOD, SVHN, and noise. 

* **Evaluation Caveats:** I must note two conditions—I initially picked the “best” method using test AUROC, and the FPR@95 column uses a test-derived threshold, meaning it isn’t directly comparable to the validation-based rejection rates. 
* **The Cost of Abstention:** The operational trade-off of rejection is that about 6% of known images are refused; however, filtering these out raises classifier accuracy on accepted known images from 95.4% to 97.5%.

---

### Q2: Describe one important design decision you made, and why did you choose that approach over alternatives?

I added a feature-space detector (`KNN`) alongside the logit-based scores. Methods like `MSP`, `Entropy`, and `Energy` are all functions of the same six final-layer numbers, so comparing them isn’t really comparing different underlying families of ideas. 

For the `KNN` approach, my hypothesis (though not directly tested) is that the 6-dimensional logits discard information that the 512-dimensional features keep. By stepping back right before the final linear layer, I extract the 512-dimensional penultimate feature vector, which contains much richer geometric information about the image's shapes and textures. 

* **Implementation Details:** I took the L2-normalized 512-d average-pool features, measured the distance to the 10th nearest neighbor ($k=10$) in a bank of 5,000 training images, and thresholded at the 5th percentile of validation scores. 
* **Performance Impact:** This improved near-OOD AUROC to 0.887 (compared to 0.859 for the best logit score) and matched `Energy` on SVHN (84.6% rejection).

---

### Q3: What is one limitation or failure case your current implementation does not handle well?

The system struggles with corrupted images. In a Gaussian-noise severity sweep, accuracy fell from 95.4% to 38.8% at $\sigma = 20$, while ECE rose from 0.015 to 0.50, meaning the model stayed overconfident while mostly wrong. 

* **Detector Response:** `Energy` rejected only 51% of those corrupted images. `KNN` rejected 76%, which is better, but it flags a noisy photo of a known class as “unknown.” While this is the right action, it is for the wrong reason: the model cannot tell a degraded image from an unfamiliar object. 
* **Unaddressed Gaps:** I haven’t built anything to separate those cases (such as a simple blur or noise quality check), so the system’s stated reason for a rejection isn't entirely reliable. Additionally, I only swept Gaussian noise; blur and JPEG corruptions were tested at a single severity level each.

---

### Q4: If you had another week to work on the project, what would you improve first and why?

If I had another week, I would make three primary improvements in this order:

1. **Repeated Random Splits:** I’d repeat everything over about five random known/unknown class splits and record mean $\pm$ std. All my conclusions currently come from one split where the unseen animals look similar to some known classes; I need to know whether `KNN` really beats `Energy` or whether that depends heavily on which classes were held out. It is the cheapest way to verify if my results are robust.
2. **Frozen Backbone Evaluation:** After that, I’d run `KNN` on the frozen ImageNet features to check whether the backbone had already seen my “unknown” classes during pretraining.
3. **Seed Control:** Ensure that all random states are strictly bound to guarantee complete reproducibility across test runs.