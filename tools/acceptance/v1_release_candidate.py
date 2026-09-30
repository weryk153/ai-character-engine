from pathlib import Path
from ai_character_engine.release_candidate import evaluate_release_candidate


def main() -> None:
    # An example is not evidence that pytest or any other acceptance actually ran.
    report = evaluate_release_candidate(Path(__file__).resolve().parents[2])
    print(f"ready_for_v1={report.ready_for_v1}")
    for gate in report.blockers:
        print(f"{gate.status.value}: {gate.name} - {gate.detail}")


if __name__ == "__main__":
    main()
