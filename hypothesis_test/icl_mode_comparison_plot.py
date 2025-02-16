import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_comparison_plots(acc1, acc2, languages=None, save_path=None):
	"""
	Creates a comprehensive visualization for paired sample comparison

	Parameters:
	acc1 (list): Accuracy scores from template 1
	acc2 (list): Accuracy scores from template 2
	languages (list): Optional list of language names
	save_path (str): Optional path to save the figure
	"""
	if languages is None:
		languages = [f'Lang {i + 1}' for i in range(len(acc1))]

	# Convert to numpy arrays
	acc1 = np.array(acc1)
	acc2 = np.array(acc2)
	differences = acc2 - acc1

	# Create figure with subplots
	fig = plt.figure(figsize=(15, 10))
	gs = plt.GridSpec(2, 2)

	# 1. Paired line plot
	ax1 = fig.add_subplot(gs[0, 0])
	x = range(len(acc1))
	ax1.plot(x, acc1, 'o-', label='Template 1', color='blue', alpha=0.7)
	ax1.plot(x, acc2, 'o-', label='Template 2', color='red', alpha=0.7)
	ax1.set_xticks(x)
	ax1.set_xticklabels(languages, rotation=45, ha='right')
	ax1.set_ylabel('Accuracy (%)')
	ax1.set_title('Paired Comparison Across Languages')
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	# 2. Box plot
	ax2 = fig.add_subplot(gs[0, 1])
	box_data = [acc1, acc2]
	ax2.boxplot(box_data, labels=['Template 1', 'Template 2'])
	ax2.set_ylabel('Accuracy (%)')
	ax2.set_title('Distribution of Accuracies')
	ax2.grid(True, alpha=0.3)

	# 3. Scatter plot with diagonal line
	ax3 = fig.add_subplot(gs[1, 0])
	min_val = min(min(acc1), min(acc2))
	max_val = max(max(acc1), max(acc2))
	ax3.scatter(acc1, acc2, alpha=0.7)
	ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
	ax3.set_xlabel('Template 1 Accuracy (%)')
	ax3.set_ylabel('Template 2 Accuracy (%)')
	ax3.set_title('Template 1 vs Template 2')
	# Add text annotations for languages
	for i, lang in enumerate(languages):
		ax3.annotate(
			lang, (acc1[i], acc2[i]), xytext=(5, 5),
			textcoords='offset points', fontsize=8
			)
	ax3.grid(True, alpha=0.3)

	# 4. Difference histogram
	ax4 = fig.add_subplot(gs[1, 1])
	sns.histplot(differences, kde=True, ax=ax4)
	ax4.axvline(x=0, color='r', linestyle='--', alpha=0.5)
	ax4.set_xlabel('Difference (Template 2 - Template 1)')
	ax4.set_ylabel('Count')
	ax4.set_title('Distribution of Differences')

	# Add mean difference line
	mean_diff = np.mean(differences)
	ax4.axvline(
		x=mean_diff, color='g', linestyle='-', alpha=0.5,
		label=f'Mean diff: {mean_diff:.2f}'
		)
	ax4.legend()

	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')

	return fig


# Example usage:
languages = ['bg', 'da', 'de', 'en', 'et', 'fa', 'fr', 'hr', 'it', 'ja', 'ko', 'nl', 'zh']
acc_template1 = [54.87, 66.41, 59.49, 67.69, 54.36, 65.13, 59.23, 55.13, 54.36, 54.36, 55.64, 56.67, 64.10]
acc_template2 = [56.67, 60.26, 63.85, 64.62, 52.82, 67.95, 59.23, 56.67, 58.72, 57.95, 57.18, 62.56, 60.77]

fig = create_comparison_plots(acc_template1, acc_template2, languages)
plt.show()
