const API_BASE = window.location.origin.includes("5000") ? window.location.origin : "http://127.0.0.1:5000";

const moods = [
  { name: "Happy", icon: "😊", desc: "Bright and positive" },
  { name: "Energetic", icon: "⚡", desc: "Fast and powerful" },
  { name: "Calm", icon: "🌿", desc: "Soft and peaceful" },
  { name: "Romantic", icon: "💚", desc: "Warm and emotional" },
  { name: "Angry", icon: "🔥", desc: "Intense and loud" },
  { name: "Low Energy", icon: "🌙", desc: "Slow and sleepy" },
  { name: "Sad", icon: "🌧️", desc: "Deep and mellow" },
];

const fallbackRecommendations = {
  Happy: [
    { title: "Golden Hour Drive", artist: "Nova Lane", score: 98 },
    { title: "Sunshine Replay", artist: "The Brights", score: 95 },
    { title: "Good Day Loop", artist: "Milo Verse", score: 92 },
    { title: "Weekend Smile", artist: "Ari Cloud", score: 90 },
    { title: "Feel Good Again", artist: "Juno Parks", score: 88 },
  ],
  Energetic: [
    { title: "Neon Sprint", artist: "Pulse Unit", score: 99 },
    { title: "Run The Night", artist: "Vexa", score: 96 },
    { title: "Electric Crowd", artist: "DJ Orbit", score: 94 },
    { title: "No Brakes", artist: "Kai Motion", score: 91 },
    { title: "Bassline Rush", artist: "Hyper City", score: 89 },
  ],
  Calm: [
    { title: "Quiet Window", artist: "Luna Shore", score: 97 },
    { title: "Soft Rain Walk", artist: "Eden Vale", score: 94 },
    { title: "Low Tide", artist: "Moss Theory", score: 92 },
    { title: "Breathe Slowly", artist: "River North", score: 90 },
    { title: "Cloud Room", artist: "Nia Field", score: 87 },
  ],
  Romantic: [
    { title: "Only With You", artist: "Sora Blue", score: 98 },
    { title: "Slow Dance Signal", artist: "Mia Vale", score: 96 },
    { title: "Paper Hearts", artist: "Eli Moon", score: 93 },
    { title: "Warm Lights", artist: "The Velvet Days", score: 91 },
    { title: "Close To Home", artist: "June Atlas", score: 88 },
  ],
  Angry: [
    { title: "Red Static", artist: "Iron Echo", score: 99 },
    { title: "Break The Wall", artist: "Riot Frame", score: 96 },
    { title: "Heavy Signal", artist: "Noir Engine", score: 94 },
    { title: "Sharp Edge", artist: "Raze Bloom", score: 91 },
    { title: "Burn Mode", artist: "Crash Theory", score: 88 },
  ],
  "Low Energy": [
    { title: "Midnight Blanket", artist: "Noah Grey", score: 97 },
    { title: "Half Awake", artist: "Velvet Fog", score: 95 },
    { title: "Dim Lamp", artist: "Sage Motel", score: 92 },
    { title: "Slow Orbit", artist: "Blue Finch", score: 89 },
    { title: "After Hours Drift", artist: "Mellow Kin", score: 87 },
  ],
  Sad: [
    { title: "Empty Station", artist: "Clara West", score: 98 },
    { title: "Rain On Tuesday", artist: "Northline", score: 95 },
    { title: "Faded Letters", artist: "Aster Grey", score: 93 },
    { title: "Last Train Home", artist: "Hollow June", score: 90 },
    { title: "Blue Apartment", artist: "The Still Room", score: 88 },
  ],
};

const moodGrid = document.getElementById("moodGrid");
const selectedMood = document.getElementById("selectedMood");
const matchScore = document.getElementById("matchScore");
const scoreLabel = document.getElementById("scoreLabel");
const songList = document.getElementById("songList");
const songCount = document.getElementById("songCount");
const topTrack = document.getElementById("topTrack");
const topArtist = document.getElementById("topArtist");
const albumArt = document.getElementById("albumArt");
const searchInput = document.getElementById("searchInput");
const sortDropdown = document.getElementById("sortDropdown");
const sortTrigger = document.getElementById("sortTrigger");
const sortLabel = document.getElementById("sortLabel");
const shuffleBtn = document.getElementById("shuffleBtn");
const predictBtn = document.getElementById("predictBtn");
const apiStatus = document.getElementById("apiStatus");
const sourceLabel = document.getElementById("sourceLabel");
const refreshBtn = document.getElementById("refreshBtn");
const recommendCountSlider = document.getElementById("recommendCountSlider");
const recommendCountValue = document.getElementById("recommendCountValue");
const prioritizeNewSongs = document.getElementById("prioritizeNewSongs");

let currentMood = "Happy";
let currentConfidence = 0.94;
let currentSongs = fallbackRecommendations[currentMood];
let recommendationSource = "fallback";
let backendOnline = false;
let currentSort = "match";
let recommendationLimit = Number(recommendCountSlider?.value || 30);

const HISTORY_KEY = "moodifyRecentlyShownV2";
const REPEAT_CHANCE = 0.33;
const PRIORITIZE_KEY = "moodifyPrioritizeNewSongs";

function songKey(song) {
  return `${String(song.title || "").trim().toLowerCase()}::${String(song.artist || "").trim().toLowerCase()}`;
}

function getRecentlyShown(mood) {
  try {
    const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "{}");
    return Array.isArray(history[mood]) ? history[mood] : [];
  } catch {
    return [];
  }
}

function rememberShown(mood, songs) {
  try {
    const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "{}");
    const previous = Array.isArray(history[mood]) ? history[mood] : [];
    const next = [...new Set([...songs.map(songKey), ...previous])].slice(0, 250);
    history[mood] = next;
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    // Ignore private browsing / blocked localStorage cases.
  }
}

function buildMoodButtons() {
  moodGrid.innerHTML = moods.map((mood) => `
    <button class="mood-card ${mood.name === currentMood ? "active" : ""}" data-mood="${mood.name}">
      <span class="mood-icon">${mood.icon}</span>
      <span class="mood-name">${mood.name}</span>
      <span class="mood-desc">${mood.desc}</span>
    </button>
  `).join("");

  document.querySelectorAll(".mood-card").forEach((button) => {
    button.addEventListener("click", async () => {
      currentMood = button.dataset.mood;
      searchInput.value = "";
      await loadRecommendations(currentMood);
    });
  });
}

function scaleToAudioFeature(value) {
  return Math.max(0, Math.min(1, Number(value) / 10));
}

function getFeaturePayload() {
  const visibleValues = {};
  document.querySelectorAll("[data-feature]").forEach((input) => {
    visibleValues[input.dataset.feature] = Number(input.value);
  });

  return {
    danceability: scaleToAudioFeature(visibleValues.danceability || 5),
    energy: scaleToAudioFeature(visibleValues.energy || 5),
    acousticness: scaleToAudioFeature(visibleValues.acousticness || 5),
    valence: scaleToAudioFeature(visibleValues.valence || 5),
    tempo: Number(visibleValues.tempo || 120),
    // Hidden/default features kept so the existing trained model still receives all required inputs.
    loudness: -60 + (scaleToAudioFeature(visibleValues.energy || 5) * 60),
    speechiness: 0.08,
    instrumentalness: 0.02,
  };
}

function formatRangeValue(feature, value) {
  if (feature === "tempo") return `${value} BPM`;
  return String(value);
}

function updateRangeLabels() {
  document.querySelectorAll("[data-feature]").forEach((input) => {
    const valueLabel = document.querySelector(`[data-value-for="${input.dataset.feature}"]`);
    const update = () => {
      if (valueLabel) valueLabel.textContent = formatRangeValue(input.dataset.feature, input.value);
      updateRangeFill(input);
    };
    update();
    input.addEventListener("input", update);
    input.addEventListener("change", update);
  });
}

function initInfoButtons() {
  document.querySelectorAll(".info-btn").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const isOpen = button.classList.contains("active");
      document.querySelectorAll(".info-btn.active").forEach((btn) => btn.classList.remove("active"));
      if (!isOpen) button.classList.add("active");
    });
  });

  document.addEventListener("click", () => {
    document.querySelectorAll(".info-btn.active").forEach((button) => button.classList.remove("active"));
  });
}

async function checkBackend() {
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    const data = await response.json();
    backendOnline = data.status === "ok" && data.model_loaded;
    if (apiStatus) {
      apiStatus.textContent = backendOnline ? `Moodify Ready • ${data.dataset_song_count || 0} songs` : "Moodify Offline";
      apiStatus.classList.toggle("online", backendOnline);
    }
  } catch {
    backendOnline = false;
    if (apiStatus) {
      apiStatus.textContent = "Moodify fallback";
      apiStatus.classList.remove("online");
    }
  }
}

async function loadFilters() {
  return;
}


function clampRecommendationLimit(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 30;
  return Math.max(20, Math.min(40, number));
}

function updateRecommendationCountLabel() {
  recommendationLimit = clampRecommendationLimit(recommendCountSlider?.value || recommendationLimit);
  if (recommendCountValue) recommendCountValue.textContent = `${recommendationLimit} songs`;
  updateRangeFill(recommendCountSlider);
}

function updateRangeFill(input) {
  if (!input) return;
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  const value = Number(input.value || min);
  const progress = max === min ? 0 : ((value - min) / (max - min)) * 100;
  input.style.setProperty("--range-progress", `${Math.max(0, Math.min(100, progress))}%`);
}

function isPrioritizingNewSongs() {
  return prioritizeNewSongs ? prioritizeNewSongs.checked : true;
}

function savePrioritizePreference() {
  try {
    localStorage.setItem(PRIORITIZE_KEY, isPrioritizingNewSongs() ? "1" : "0");
  } catch {
    // Ignore blocked localStorage cases.
  }
}

function loadPrioritizePreference() {
  if (!prioritizeNewSongs) return;
  try {
    const saved = localStorage.getItem(PRIORITIZE_KEY);
    if (saved === "0") prioritizeNewSongs.checked = false;
    if (saved === "1") prioritizeNewSongs.checked = true;
  } catch {
    // Keep default checked.
  }
}

function recommendationPayload(mood) {
  const prioritize = isPrioritizingNewSongs();
  return {
    mood,
    limit: recommendationLimit,
    avoid_keys: prioritize ? getRecentlyShown(mood) : [],
    repeat_chance: prioritize ? REPEAT_CHANCE : 1,
  };
}


async function loadRecommendations(mood) {
  setLoading(true);
  try {
    if (!backendOnline) throw new Error("Backend unavailable");
    const response = await fetch(`${API_BASE}/api/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(recommendationPayload(mood)),
    });
    if (!response.ok) throw new Error("Recommendation request failed");
    const data = await response.json();
    currentMood = data.mood;
    currentConfidence = data.confidence || 0.94;
    currentSongs = data.recommendations || fallbackRecommendations[currentMood];
    rememberShown(currentMood, currentSongs);
    recommendationSource = data.source || "model";
  } catch {
    currentMood = mood;
    currentConfidence = 0.94;
    currentSongs = fallbackRecommendations[mood];
    recommendationSource = "fallback";
  } finally {
    setLoading(false);
    render();
  }
}

async function analyzeMood() {
  predictBtn.disabled = true;
  predictBtn.textContent = "Analyzing...";
  setLoading(true);
  try {
    if (!backendOnline) await checkBackend();
    if (!backendOnline) throw new Error("Backend unavailable");
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...getFeaturePayload(),
        limit: recommendationLimit,
        avoid_keys: isPrioritizingNewSongs() ? getRecentlyShown(currentMood) : [],
        repeat_chance: isPrioritizingNewSongs() ? REPEAT_CHANCE : 1,
      }),
    });
    if (!response.ok) throw new Error("Prediction failed");
    const data = await response.json();
    currentMood = data.mood;
    currentConfidence = data.confidence || 0.9;
    currentSongs = data.recommendations || fallbackRecommendations[currentMood];
    rememberShown(currentMood, currentSongs);
    recommendationSource = data.source || "model";
  } catch (error) {
    if (apiStatus) {
      apiStatus.textContent = "Start backend first";
      apiStatus.classList.remove("online");
    }
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Analyze Mood";
    setLoading(false);
    render();
  }
}

function setLoading(isLoading) {
  songList.classList.toggle("loading", isLoading);
}

function getFilteredSongs() {
  const query = searchInput.value.trim().toLowerCase();
  const sort = currentSort;
  let songs = [...currentSongs];

  if (query) {
    songs = songs.filter((song) =>
      song.title.toLowerCase().includes(query) || song.artist.toLowerCase().includes(query)
    );
  }

  if (sort === "title") songs.sort((a, b) => a.title.localeCompare(b.title));
  if (sort === "artist") songs.sort((a, b) => a.artist.localeCompare(b.artist));
  if (sort === "match") songs.sort((a, b) => b.score - a.score);

  return songs;
}

function renderSongs() {
  const songs = getFilteredSongs();
  songCount.textContent = `${songs.length} / ${recommendationLimit} songs`;
  sourceLabel.textContent = "";

  if (!songs.length) {
    songList.innerHTML = `<div class="empty-state">No recommendations found. Try another search.</div>`;
    return;
  }

  songList.innerHTML = songs.map((song, index) => {
    const spotifyUrl = song.spotify_url || `https://open.spotify.com/search/${encodeURIComponent(`${song.title} ${song.artist}`)}`;
    const previewBlock = song.preview_url
      ? `<audio class="song-preview" controls preload="none" src="${song.preview_url}"></audio>`
      : song.spotify_embed_url
        ? `<details class="embed-preview"><summary>Preview</summary><iframe loading="lazy" src="${song.spotify_embed_url}?utm_source=generator" width="100%" height="80" frameborder="0" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe></details>`
        : `<span class="preview-unavailable">Preview unavailable</span>`;
    return `
      <article class="song-card">
        <div class="song-rank">${String(index + 1).padStart(2, "0")}</div>
        <div class="song-main">
          <h4>${song.title}</h4>
          <p>${song.artist}</p>
          <div class="song-tags">
            <span class="source-badge">${song.genre || "pop"}</span>
            <span class="source-badge">${song.language || "English"}</span>
          </div>
        </div>
        <div class="song-meta">
          <strong>${Math.round(song.score)}%</strong>
          <a class="spotify-link" href="${spotifyUrl}" target="_blank" rel="noopener">Open Spotify</a>
        </div>
        <div class="preview-area">${previewBlock}</div>
      </article>
    `;
  }).join("");
}

function renderTopSong() {
  const top = currentSongs[0];
  selectedMood.textContent = currentMood;
  matchScore.textContent = `${Math.round(currentConfidence * 100)}%`;
  scoreLabel.textContent = backendOnline ? "Mood confidence" : "Demo confidence";
  topTrack.textContent = top.title;
  topArtist.textContent = top.artist;
  albumArt.textContent = currentMood.charAt(0);
}

function render() {
  buildMoodButtons();
  renderTopSong();
  renderSongs();
}

searchInput.addEventListener("input", renderSongs);

sortTrigger.addEventListener("click", () => {
  const isOpen = sortDropdown.classList.toggle("open");
  sortTrigger.setAttribute("aria-expanded", String(isOpen));
});

sortDropdown.querySelectorAll("[data-sort]").forEach((option) => {
  option.addEventListener("click", () => {
    currentSort = option.dataset.sort;
    sortLabel.textContent = option.textContent;
    sortDropdown.querySelectorAll("[data-sort]").forEach((btn) => btn.classList.remove("selected"));
    option.classList.add("selected");
    sortDropdown.classList.remove("open");
    sortTrigger.setAttribute("aria-expanded", "false");
    renderSongs();
  });
});

document.addEventListener("click", (event) => {
  if (!sortDropdown.contains(event.target)) {
    sortDropdown.classList.remove("open");
    sortTrigger.setAttribute("aria-expanded", "false");
  }
});

refreshBtn.addEventListener("click", () => loadRecommendations(currentMood));
if (recommendCountSlider) {
  recommendCountSlider.addEventListener("input", updateRecommendationCountLabel);
  recommendCountSlider.addEventListener("change", () => loadRecommendations(currentMood));
}

if (prioritizeNewSongs) {
  prioritizeNewSongs.addEventListener("change", () => {
    savePrioritizePreference();
    loadRecommendations(currentMood);
  });
}

predictBtn.addEventListener("click", analyzeMood);
shuffleBtn.addEventListener("click", async () => {
  const randomMood = moods[Math.floor(Math.random() * moods.length)].name;
  currentMood = randomMood;
  searchInput.value = "";
  await loadRecommendations(randomMood);
});

loadPrioritizePreference();
updateRecommendationCountLabel();
updateRangeLabels();
document.querySelectorAll('input[type="range"]').forEach(updateRangeFill);
initInfoButtons();
checkBackend().then(async () => {
  await loadFilters();
  await loadRecommendations(currentMood);
});
