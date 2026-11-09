- Use when, watch, and run_controller for anything that must react quickly. A
  program sees the scene only between motions: to reach an object that moves while
  you act, track it in run_controller with fresh camera images and servo toward it,
  instead of polling its position and predicting where it will be.
