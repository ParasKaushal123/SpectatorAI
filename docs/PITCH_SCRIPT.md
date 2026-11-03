# SpectatorAI: pitch script (7 slides, about 4 minutes)

*The slides show the facts. The script tells the story. Don't read the slide; talk to the judges.*

---

## Slide 1: Title (about 25 s)

> Every day, thousands of workers do skilled work with their hands: sanding, polishing, assembling. Strap a camera to their head and you're recording exactly what a robot needs to learn: real hands doing real work.
>
> There's a catch. That footage is almost useless to a robot as it is.
>
> We built SpectatorAI to fix that. *(gesture to the image)* On the left is what the camera saw. On the right is what we pull out of it: both of the worker's hands, in 3D, correctly labelled, frame after frame.

**Cue:** pause for a moment on the 3D hands before moving on.

---

## Slide 2: The problem (about 35 s)

> We started the obvious way: off-the-shelf hand tracking on our factory clips. *(point at the top image)* Here it loses a hand that's clearly in the frame. *(point at the bottom image)* And here it confidently tracks a piece of cloth as a hand.
>
> Now picture a robot learning from that.
>
> And those are only the visible failures. Half of a real clip might be blur or the person walking around. When a hand leaves the frame, most systems either drop it or make up a position and don't tell you. Even when tracking works, every coordinate moves with the worker's head. To a robot standing still, that's noise.
>
> So the real problem isn't detecting hands. It's producing data you can trust.

---

## Slide 3: The solution (about 40 s)

> Our first insight: don't decide on every frame. A single frame can fool any model, so we're patient. We follow each candidate over time, collect evidence (how far away it is, where its arm enters the picture, what it looks like), and only then rule: left hand, right hand, or not the wearer's at all. Every rejection comes with a written reason.
>
> Second, we lift the hands into full 3D and correct for the fisheye distortion. When we have to fill a gap, we say so. Nothing invented gets passed off as measured.
>
> Third, we grade every second of the video. So a dataset team doesn't get one long file; they get exactly which seconds are worth training on, and why the rest aren't.

**Cue:** tap each of the three columns as you go.

---

## Slide 4: Our edge, the world frame (about 40 s)

> This is the part we're proudest of.
>
> *(left image)* Almost every hand tracker stops here. The hands are described relative to a camera on someone's head, and that head never stops moving. Lean forward and every number changes, even though the hands didn't move.
>
> *(right image)* We added SLAM. We rebuild the camera's path and the space around it, find which way is up from gravity and the workbench, and then place the hands back into that fixed room. That small frustum is the worker's head. The yellow line is where they're looking.
>
> The result: a hand resting on the bench stays on the bench. Now a robot can use it, because a robot lives in the room, not inside someone's head.

---

## Slide 5: Architecture (about 35 s)

> Here's how it fits together, end to end.
>
> Video goes in on the left. Four stages run on a cloud GPU: hand tracking, quality rating, the world view, and a 3D twin of the workstation built from two separate recordings.
>
> *(point to the right panel)* We designed for the real world. SLAM fails on hard footage, so instead of crashing we retry with safer settings, and as a last resort we still give you head rotation. Every clip produces an answer.
>
> And everything is cached, so if the cloud session drops halfway through, we pick up where we left off.
>
> We stand on strong open models: WiLoR, CLIP, NVIDIA's ViPE, MoGe. What's ours is the decision-making that turns them into one reliable pipeline.

**Cue:** skim it. Don't read the boxes, just follow the arrow down.

---

## Slide 6: Prototype (about 30 s)

> This isn't a single cherry-picked demo. These are four different real stations: woodwork, assembly, materials handling, polishing. Different lighting, different motions, different clutter. Same pipeline, no per-station tuning.
>
> We ran it across forty real factory clips. And notice the bottom left: when a hand leaves the frame, we don't fake it; it's shown as out of view.
>
> *(if there's time)* We've also gone one step further: we replayed these motions on two virtual robot arms and scored, frame by frame, how much of the task a robot could physically copy. On one two-minute clip, that was about eighty percent.

---

## Slide 7: Conclusion (about 30 s)

> So here's where we are.
>
> Today, SpectatorAI takes raw head-camera footage and gives you the wearer's hands, in 3D, inside a real room, with the good seconds already picked out.
>
> Next, we scale that into large, searchable libraries of human demonstrations across many tasks, and feed them straight into robot learning.
>
> The knowledge of how physical work gets done already exists; it's in the hands of skilled people. We built the bridge that lets robots learn from it.
>
> Thank you. We'd love your questions.

---

## Likely judge questions (short answers)

- **"Is the hand model yours?"** The 3D hand model is pretrained WiLoR. Our contribution is everything around it: deciding which hand is the wearer's, keeping left and right consistent, fixing the fisheye, labelling gaps, rating quality, and placing everything in the room.
- **"How accurate is it?"** On synthetic tests with known answers: about 99% hand recall and 100% correct left/right, with hands placed within about a centimetre in the room. On the 40 real clips we checked the hardest cases by eye. Measuring against hand-labelled frames is our next step.
- **"Why SLAM, if the hands already looked right?"** SLAM doesn't change the hand shape. It changes where the hand is: in a fixed, level room, in metres. That's what a robot needs, and it's what makes the 3D room and the robot replay possible.
- **"How long does it take?"** Full quality is about 25 minutes per 3-minute clip on one cloud GPU, with four clips in parallel. Our web app has a fast mode aimed at a 30-second clip in one to two minutes.
- **"What about privacy?"** Footage stays private. Nothing is published, the access key never leaves the local machine, and all copies are deleted after the event.

## Delivery tips

- Lead with the human story (skilled hands, robots learning), not the technology.
- On slides 4 and 6, let the visuals land; silence is fine.
- If you're short on time, cut the robot line on slide 6 and the caching line on slide 5.
