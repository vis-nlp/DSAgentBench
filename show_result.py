import argparse
import json
import os


def get_result(action_space, use_model, observation_type, result_dir):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return None

    all_result = []
    domain_result = {}
    all_result_for_analysis = {}

    for domain in sorted(os.listdir(target_dir)):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in sorted(os.listdir(domain_path)):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    result_file = os.path.join(example_path, "result.txt")
                    if os.path.exists(result_file):
                        if domain not in domain_result:
                            domain_result[domain] = []
                        with open(result_file, "r") as rf:
                            raw_score = rf.read().strip()
                        try:
                            score = float(raw_score)
                        except Exception:
                            try:
                                score = float(eval(raw_score))
                            except Exception:
                                score = 0.0

                        domain_result[domain].append(score)
                        all_result.append(score)

                        if domain not in all_result_for_analysis:
                            all_result_for_analysis[domain] = {}
                        all_result_for_analysis[domain][example_id] = score

    if not domain_result:
        print(f"No results found under {target_dir}")
        return None

    print("\n" + "=" * 50)
    print(f"Evaluation Results: {use_model} ({observation_type})")
    print("=" * 50)
    for domain, scores in domain_result.items():
        avg_rate = (sum(scores) / len(scores)) * 100 if scores else 0.0
        print(f"Domain: {domain:<20} | Completed: {len(scores):<5} | Success Rate: {avg_rate:.2f}%")

    # Optional OSWorld aggregate groups if present
    office_domains = ["libreoffice_calc", "libreoffice_impress", "libreoffice_writer"]
    office_scores = [s for d in office_domains if d in domain_result for s in domain_result[d]]
    if office_scores:
        print(f"Office Suite        | Completed: {len(office_scores):<5} | Success Rate: {(sum(office_scores)/len(office_scores))*100:.2f}%")

    daily_domains = ["vlc", "thunderbird", "chrome"]
    daily_scores = [s for d in daily_domains if d in domain_result for s in domain_result[d]]
    if daily_scores:
        print(f"Daily Apps          | Completed: {len(daily_scores):<5} | Success Rate: {(sum(daily_scores)/len(daily_scores))*100:.2f}%")

    prof_domains = ["gimp", "vs_code"]
    prof_scores = [s for d in prof_domains if d in domain_result for s in domain_result[d]]
    if prof_scores:
        print(f"Professional Apps   | Completed: {len(prof_scores):<5} | Success Rate: {(sum(prof_scores)/len(prof_scores))*100:.2f}%")

    print("-" * 50)
    overall_rate = (sum(all_result) / len(all_result)) * 100 if all_result else 0.0
    print(f"Overall Total       | Completed: {len(all_result):<5} | Success Rate: {overall_rate:.2f}%")
    print("=" * 50 + "\n")

    summary_json_path = os.path.join(target_dir, "all_result.json")
    with open(summary_json_path, "w") as f:
        json.dump(all_result_for_analysis, f, indent=2)

    return all_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Display and aggregate DSWorld / OSWorld benchmark results.")
    parser.add_argument("--action_space", type=str, default="pyautogui", help="Action space used in evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name evaluated")
    parser.add_argument("--observation_type", type=str, default="screenshot_a11y_tree", help="Observation type used")
    parser.add_argument("--result_dir", type=str, default="./results_data_science", help="Base results directory")

    args = parser.parse_args()
    get_result(args.action_space, args.model, args.observation_type, args.result_dir)
