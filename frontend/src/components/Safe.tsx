import { Component, type ReactNode } from "react";

/** Keeps one broken section from blanking the whole page during a demo. */
export default class Safe extends Component<{ name: string; children: ReactNode }, { err: string | null }> {
  state = { err: null as string | null };
  static getDerivedStateFromError(e: unknown) {
    return { err: String(e) };
  }
  render() {
    if (!this.state.err) return this.props.children;
    return (
      <div className="wrap py-27">
        <div className="card-soft bg-rose-wash text-body-sm">
          {this.props.name} couldn’t render: {this.state.err}{" "}
          <button className="btn-ghost ml-9 px-13 py-6" onClick={() => this.setState({ err: null })}>Retry</button>
        </div>
      </div>
    );
  }
}
