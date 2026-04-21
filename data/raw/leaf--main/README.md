# LEAF++: Enriching Essay Feedback Dataset with Trait Scores

## Dataset description
The LEAF++ dataset is an extended version of the original LEAF dataset by Behzad et al. (2024)([Paper](https://aclanthology.org/2024.naacl-short.36.pdf), [Dataset](https://github.com/shabnam-b/LEAF)), which previously included only free-form feedback. LEAF++ enriches the original dataset with detailed trait-level annotations, providing scores across multiple dimensions. Here are the names for each traits:
- <code>trait_1</code>: Grammar accuracy
- <code>trait_2</code>: Appropriateness of word use
- <code>trait_3</code>: Elasticity of sentence expression
- <code>trait_4</code>: Appropriateness of structure within a paragraph
- <code>trait_5</code>: Adequacy of inter-paragraph structure
- <code>trait_6</code>: Consistency of structure
- <code>trait_7</code>: Appropriateness of portion size
- <code>trait_8</code>: Clarity of topic
- <code>trait_9</code>: Specificity of explanation
- <code>trait_10</code>: Creativity of thought

## LEAF-Feedback-Dimension
The <code>LEAF-Feedback-Dimension</code> folder contains the same dataset with the trait scores are aligned with the feedback dimensions in the dataset. The alignment is done by averaging traits from the LEAF++. In addition, the essays are preprocessed. 
- <code>alignment_with_topic</code>: This is for alignment with topic (prompt adherence) issues. The score is the average of clarity of topic, specificity of explanation and creativity of thought.
- <code>spelling_grammar_style</code>: This is for spelling, grammar, and style issues. The score is the average of grammar accuracy, appropriateness of word use, and elasticity of sentence expression.
- <code>clarity_of_view_point</code>: This is for the clarity of author's point of view issues. The score is the average of appropriateness of structure within a paragraph, adequency of inter-paragraph structure, and consistency of structure.
- <code>arguments_supporting_datails</code>: This is for arguments and supporting details issues. The score is the average of appropraiteness of portion size, and specificity of explanation.
