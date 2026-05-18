import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import './App.css';

const MAX_DIVERSE_COUNT = 7;

const formatArticleDate = (date) => {
  if (!date) return '';

  const parsedDate = new Date(date);
  if (Number.isNaN(parsedDate.valueOf())) return date;

  return parsedDate.toLocaleDateString('sl-SI', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });
};

const getSimilarityColor = (score = 0) => {
  const normalizedScore = Math.max(0, Math.min(1, Number(score) || 0));
  const hue = 12 + normalizedScore * 135;

  return `hsl(${hue} 72% 45%)`;
};

const formatSimilarityScore = (score = 0) => (
  `${(Math.max(0, Math.min(1, Number(score) || 0)) * 100).toFixed(1)}%`
);

const getInitialTheme = () => {
  const storedTheme = window.localStorage.getItem('article-theme');
  if (storedTheme === 'dark' || storedTheme === 'light') return storedTheme;

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

const formatEntities = (entities = []) => (
  Array.isArray(entities) && entities.length > 0 ? entities.join(', ') : 'Ni zaznanih entitet'
);

const ArticleCard = React.forwardRef(function ArticleCard(
  { article, isSelected, isDimmed, onClick },
  ref,
) {
  return (
    <article
      ref={ref}
      className={`article-card ${isSelected ? 'selected' : ''} ${isDimmed ? 'dimmed' : ''}`}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
      role="button"
      tabIndex={0}
      aria-pressed={isSelected}
    >
      {article.thumbnail && (
        <img src={article.thumbnail} alt="" className="article-thumb" />
      )}
      <div className="article-content">
        <h3>{article.title}</h3>
        <span className="article-date">{article.date}</span>
        {article.lead && <p className="article-lead">{article.lead}</p>}
        {article.score > 0 ? (
          <span className="article-score">Ujemanje: {(article.score * 100).toFixed(2)}%</span>
        ) : null}
      </div>
    </article>
  );
});

const RouteSummaryBox = ({ summary, isLoading, isChainLoading, llmEnabled }) => (
  <aside className="route-summary-box">
    <strong>Skupna povezava</strong>
    {isChainLoading ? (
      <p>Nalagam celotno verigo...</p>
    ) : !llmEnabled ? (
      <p>LLM povezave so izklopljene.</p>
    ) : isLoading && !summary ? (
      <p>Generiram razlago...</p>
    ) : summary ? (
      <>
        <p>{summary.connection}</p>
        <small>{formatEntities(summary.entities)}</small>
      </>
    ) : (
      <p>Razlaga se bo prikazala tukaj.</p>
    )}
  </aside>
);

const RoutePreview = React.forwardRef(function RoutePreview(
  { article, onClick, onTimelessChain, isTimelessLoading },
  ref,
) {
  const isInteractive = Boolean(onClick);

  return (
    <div className="route-preview-wrap">
      <aside 
        className={`route-preview ${isInteractive ? 'interactive' : ''}`} 
        aria-label="Izbran clanek" 
        ref={ref}
        onClick={(e) => {
          if (isInteractive) {
            e.stopPropagation();
            onClick();
          }
        }}
        role={isInteractive ? "button" : undefined}
        tabIndex={isInteractive ? 0 : undefined}
      >
        {article.thumbnail ? (
          <img src={article.thumbnail} alt="" />
        ) : (
          <div className="route-preview-placeholder" />
        )}
        <h3>{article.title}</h3>
      </aside>
      <time className="route-node-date">{formatArticleDate(article.date)}</time>
      {onTimelessChain && (
        <button
          type="button"
          className="timeless-chain-button"
          onClick={(e) => {
            e.stopPropagation();
            onTimelessChain();
          }}
          disabled={isTimelessLoading}
        >
          <span>{isTimelessLoading ? 'Gradim verigo...' : 'Veriga brez casovne omejitve'}</span>
          <small>Ce imajo vasi clanki slabo podobnost originalnemu in je datum v letu 2026 uporabite ta gumb</small>
        </button>
      )}
    </div>
  );
});

const RouteChildNode = React.forwardRef(function RouteChildNode(
  { article, isHovered, isSelected, onClick, onLinkClick, onHoverChange, showScore = true, showText = true },
  ref,
) {
  const branchColor = getSimilarityColor(article.score);
  const isInteractive = Boolean(onClick);
  const hasArticleLink = Boolean(article.url || article.link);

  return (
    <div
      className={`route-child-wrap ${isHovered ? 'hovered' : ''} ${isSelected ? 'selected' : ''}`}
      style={{ '--branch-color': branchColor }}
      onMouseEnter={() => onHoverChange(article.id)}
      onMouseLeave={() => onHoverChange(null)}
      onFocus={() => onHoverChange(article.id)}
      onBlur={() => onHoverChange(null)}
    >
      {showScore && (
        <div className="route-score-badge">
          Ujemanje {formatSimilarityScore(article.score)}
        </div>
      )}
      <article
        ref={ref}
        className={`route-child-node ${isInteractive ? 'interactive' : ''}`}
        onClick={(e) => {
          if (isInteractive) {
            e.stopPropagation();
            onClick();
          }
        }}
        onKeyDown={(e) => {
          if (isInteractive && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            e.stopPropagation();
            onClick();
          }
        }}
        role={isInteractive ? 'button' : undefined}
        tabIndex={isInteractive ? 0 : undefined}
        aria-pressed={isInteractive ? isSelected : undefined}
      >
        {article.thumbnail ? (
          <img src={article.thumbnail} alt="" />
        ) : (
          <div className="route-child-placeholder" />
        )}
        {showText && <h3>{article.title}</h3>}
      </article>
      {showText && <time className="route-node-date">{formatArticleDate(article.date)}</time>}
      {showText && hasArticleLink && (
        <button
          type="button"
          className="route-article-link"
          onClick={(e) => {
            e.stopPropagation();
            onLinkClick();
          }}
        >
          Povezava do članka
        </button>
      )}
    </div>
  );
});

const RouteFirstLinkInfo = ({ article }) => (
  <div className="route-first-link-info">
    <h3>{article.title}</h3>
    <div className="route-first-link-meta">
      <time>{formatArticleDate(article.date)}</time>
      <span>Ujemanje {formatSimilarityScore(article.score)}</span>
    </div>
  </div>
);

const RouteTree = ({
  rootArticle,
  childArticles,
  isLoading,
  selectedChainId,
  onChainClick,
  isFetchingSummaries,
  llmEnabled,
  onArticleRestart,
  onTimelessChain,
  isTimelessLoading,
  onBack,
}) => {
  const stageRef = useRef(null);
  const rootRef = useRef(null);
  const nodeRefs = useRef({});
  const [branches, setBranches] = useState([]);
  const [hoveredChildId, setHoveredChildId] = useState(null);
  const routeRows = useMemo(
    () => childArticles.map((article, rowIndex) => {
      const chainArticles = [article, ...(article.chain || [])];
      const isSelected = selectedChainId === article.id;
      const visibleArticles = isSelected ? [rootArticle, ...chainArticles] : [article];

      return {
        chainId: article.id,
        isSelected,
        summary: article.route_summary,
        chainReady: Boolean(article.chain_ready),
        summaryReady: Boolean(article.summary_ready),
        nodes: visibleArticles.map((routeArticle, nodeIndex) => ({
          article: routeArticle,
          isRoot: isSelected && nodeIndex === 0,
          key: `${rowIndex}-${nodeIndex}-${routeArticle.id}`,
        })),
      };
    }),
    [childArticles, rootArticle, selectedChainId],
  );
  const rowCount = Math.max(routeRows.length, 1);
  const childGap = 9;
  const chainGap = 36;

  useLayoutEffect(() => {
    const measureBranches = () => {
      if (!stageRef.current || routeRows.length === 0) {
        setBranches([]);
        return;
      }

      const stageBox = stageRef.current.getBoundingClientRect();
      const rootBox = rootRef.current?.getBoundingClientRect();

      const createBranch = (fromBox, toBox, targetArticle, chainId) => {
        const startX = fromBox.right - stageBox.left;
        const startY = fromBox.top + fromBox.height / 2 - stageBox.top;
        const endX = toBox.left - stageBox.left;
        const endY = toBox.top + toBox.height / 2 - stageBox.top;
        const distance = Math.max(32, endX - startX);
        const curve = Math.min(120, Math.max(28, distance * 0.42));

        return {
          articleId: targetArticle.id,
          chainId,
          color: getSimilarityColor(targetArticle.score),
          d: `M ${startX} ${startY} C ${startX + curve} ${startY}, ${endX - curve} ${endY}, ${endX} ${endY}`,
        };
      };

      const nextBranches = [];

      routeRows.forEach((row) => {
        const firstNode = nodeRefs.current[row.nodes[0]?.key];
        if (!firstNode) return;

        if (!selectedChainId && rootBox) {
          nextBranches.push(
            createBranch(rootBox, firstNode.getBoundingClientRect(), row.nodes[0].article, row.chainId),
          );
        }

        for (let index = 1; index < row.nodes.length; index += 1) {
          const previousNode = nodeRefs.current[row.nodes[index - 1].key];
          const currentNode = nodeRefs.current[row.nodes[index].key];
          if (!previousNode || !currentNode) continue;

          nextBranches.push(
            createBranch(
              previousNode.getBoundingClientRect(),
              currentNode.getBoundingClientRect(),
              row.nodes[index].article,
              row.chainId,
            ),
          );
        }
      });

      setBranches(nextBranches);
    };

    measureBranches();
    const frame = window.requestAnimationFrame(() => {
      measureBranches();
      window.requestAnimationFrame(measureBranches);
    });
    const settledTimer = window.setTimeout(measureBranches, 320);
    window.addEventListener('resize', measureBranches);

    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(settledTimer);
      window.removeEventListener('resize', measureBranches);
    };
  }, [routeRows, selectedChainId]);

  return (
    <div className="route-stage" ref={stageRef}>
      <button className="route-back-button" onClick={onBack} aria-label="Nazaj">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M19 12H5M12 19l-7-7 7-7"/>
        </svg>
        Nazaj
      </button>

      {!selectedChainId && (
        <RoutePreview 
          article={rootArticle} 
          ref={rootRef} 
          onClick={(rootArticle.url || rootArticle.link) ? () => window.open(rootArticle.url || rootArticle.link, '_blank') : undefined} 
          onTimelessChain={onTimelessChain}
          isTimelessLoading={isTimelessLoading}
        />
      )}

      <svg className="route-branches" aria-hidden="true">
        {branches.map((branch, index) => (
          <path
            key={`${branch.d}-${index}`}
            className={`${branch.chainId === selectedChainId ? 'selected' : ''} ${selectedChainId && branch.chainId !== selectedChainId ? 'dimmed' : ''} ${branch.articleId === hoveredChildId ? 'hovered' : ''}`}
            d={branch.d}
            style={{ '--branch-color': branch.color }}
          />
        ))}
      </svg>

      <div
        className={`route-tree-rows ${selectedChainId ? 'has-selected-chain' : ''}`}
        style={{
          '--child-gap': `${childGap}px`,
          '--chain-gap': `${chainGap}px`,
          '--child-size': `clamp(38px, calc((100svh - 158px - ${(rowCount - 1) * childGap}px) / ${rowCount} - 20px), 140px)`,
          '--focused-child-size': 'clamp(120px, 18svh, 170px)',
        }}
      >
        {isLoading ? (
          <div className="loading">Pridobivanje raznolikih clankov...</div>
        ) : (
          routeRows.map((row) => (
            <div
              className={`route-tree-row ${row.isSelected ? 'selected-chain' : ''} ${selectedChainId && !row.isSelected ? 'dimmed-chain' : ''}`}
              key={row.chainId}
              onClick={() => onChainClick(row.chainId)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onChainClick(row.chainId);
                }
              }}
              role="button"
              tabIndex={0}
              aria-pressed={row.isSelected}
            >
              {row.isSelected && (
                <RouteSummaryBox
                  summary={row.summary}
                  isLoading={isFetchingSummaries && !row.summaryReady}
                  isChainLoading={!row.chainReady}
                  llmEnabled={llmEnabled}
                />
              )}
              <div className="route-chain-nodes">
              {row.nodes.map(({ article, key, isRoot }) => (
                <RouteChildNode
                  key={key}
                  ref={(node) => {
                    if (node) {
                      nodeRefs.current[key] = node;
                    } else {
                      delete nodeRefs.current[key];
                    }
                  }}
                  article={article}
                  isHovered={article.id === hoveredChildId}
                  isSelected={row.isSelected}
                  onHoverChange={setHoveredChildId}
                  showScore={!isRoot}
                  showText={row.isSelected}
                  onClick={row.isSelected ? () => onArticleRestart(article) : undefined}
                  onLinkClick={() => window.open(article.url || article.link, '_blank')}
                />
              ))}
                {!row.isSelected && (
                  <RouteFirstLinkInfo article={row.nodes[0].article} />
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

function App() {
  const [theme, setTheme] = useState(getInitialTheme);
  const [query, setQuery] = useState('');
  const [k, setK] = useState(20);
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isRandomLoading, setIsRandomLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [selectedArticle, setSelectedArticle] = useState(null);
  const [diverseCount, setDiverseCount] = useState(5);
  const [isFetchingDiverse, setIsFetchingDiverse] = useState(false);
  const [isFetchingTimelessChain, setIsFetchingTimelessChain] = useState(false);
  const [diverseResults, setDiverseResults] = useState([]);
  const [error, setError] = useState('');
  const [isPinnedPreview, setIsPinnedPreview] = useState(false);
  const [isClearingResults, setIsClearingResults] = useState(false);
  const [selectedChainId, setSelectedChainId] = useState(null);
  const [useLlmOutputs, setUseLlmOutputs] = useState(true);
  const [isFetchingSummaries, setIsFetchingSummaries] = useState(false);
  const [chainJobId, setChainJobId] = useState(null);
  const selectedAnchorRef = useRef(null);
  const pinTimerRef = useRef(null);

  const orderedResults = useMemo(() => {
    if (!selectedArticle) return results;

    return [
      selectedArticle,
      ...results.filter((article) => article.id !== selectedArticle.id),
    ];
  }, [results, selectedArticle]);

  useEffect(() => {
    if (!selectedArticle || isPinnedPreview) return;

    window.requestAnimationFrame(() => {
      selectedAnchorRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
    });
  }, [selectedArticle, isPinnedPreview]);

  useEffect(() => {
    document.body.classList.toggle('route-lock-scroll', isPinnedPreview);
    document.documentElement.classList.toggle('route-lock-scroll', isPinnedPreview);

    return () => {
      document.body.classList.remove('route-lock-scroll');
      document.documentElement.classList.remove('route-lock-scroll');
    };
  }, [isPinnedPreview]);

  useEffect(() => () => window.clearTimeout(pinTimerRef.current), []);

  useEffect(() => {
    if (!chainJobId) return undefined;

    let isCancelled = false;
    let pollTimer = null;

    const pollChainJob = async () => {
      try {
        const response = await fetch(`http://127.0.0.1:8000/api/chain-jobs/${chainJobId}`);
        if (!response.ok) throw new Error('Chain job polling failed');
        const data = await response.json();
        if (isCancelled) return;

        setDiverseResults(data.results || []);
        setIsFetchingSummaries(data.status === 'loading_summaries');

        if (data.status === 'complete' || data.status === 'error') {
          setChainJobId(null);
          setIsFetchingSummaries(false);
          return;
        }

        pollTimer = window.setTimeout(pollChainJob, 1200);
      } catch (error) {
        console.error('Error polling chain job:', error);
        if (!isCancelled) {
          pollTimer = window.setTimeout(pollChainJob, 1800);
        }
      }
    };

    pollTimer = window.setTimeout(pollChainJob, 700);

    return () => {
      isCancelled = true;
      window.clearTimeout(pollTimer);
    };
  }, [chainJobId]);

  const clearPinTimer = () => {
    window.clearTimeout(pinTimerRef.current);
    pinTimerRef.current = null;
  };

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem('article-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((currentTheme) => (
      currentTheme === 'dark' ? 'light' : 'dark'
    ));
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsSearching(true);
    setHasSearched(true);
    setError('');

    try {
      const response = await fetch(`http://127.0.0.1:8000/api/search?q=${encodeURIComponent(query)}&k=${k}`);
      if (!response.ok) throw new Error('Search request failed');
      const data = await response.json();
      setResults(data.results || []);
      setSelectedArticle(null);
      setDiverseResults([]);
      setSelectedChainId(null);
      setIsFetchingSummaries(false);
      setChainJobId(null);
      setIsPinnedPreview(false);
      setIsClearingResults(false);
    } catch (error) {
      console.error('Error fetching articles:', error);
      setError('Pri iskanju je prislo do napake.');
    } finally {
      setIsSearching(false);
    }
  };

  const handleRandomArticle = async () => {
    setIsRandomLoading(true);
    setHasSearched(true);
    setResults([]);
    setError('');
    setIsClearingResults(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/random');
      if (!response.ok) throw new Error('Random article request failed');
      const data = await response.json();
      await fetchDiverseForArticle(data.article, { pinDelay: 0 });
    } catch (error) {
      console.error('Error fetching random article:', error);
      setError('Pri izbiri naključnega članka je prišlo do napake.');
      setHasSearched(false);
      setIsClearingResults(false);
    } finally {
      setIsRandomLoading(false);
    }
  };

  const resetSearch = () => {
    setHasSearched(false);
    setResults([]);
    setQuery('');
    setSelectedArticle(null);
    setDiverseResults([]);
    setSelectedChainId(null);
    setIsFetchingSummaries(false);
    setChainJobId(null);
    setError('');
    setIsPinnedPreview(false);
    setIsClearingResults(false);
    clearPinTimer();
  };

  const handleArticleClick = (article) => {
    setSelectedArticle(article);
    setDiverseResults([]);
    setSelectedChainId(null);
    setIsFetchingSummaries(false);
    setChainJobId(null);
    setError('');
    setIsPinnedPreview(false);
    setIsClearingResults(false);
    clearPinTimer();
  };

  const fetchDiverseForArticle = async (baseArticle, { pinDelay = 180 } = {}) => {
    if (!baseArticle) return;

    setSelectedArticle(baseArticle);
    setIsFetchingDiverse(true);
    setIsFetchingSummaries(false);
    setChainJobId(null);
    setError('');
    setIsClearingResults(true);
    setSelectedChainId(null);
    setDiverseResults([]);
    const requestedCount = Math.min(diverseCount, MAX_DIVERSE_COUNT);
    clearPinTimer();
    pinTimerRef.current = window.setTimeout(() => {
      setIsPinnedPreview(true);
    }, pinDelay);

    try {
      const response = await fetch(`http://127.0.0.1:8000/api/diverse?article_id=${encodeURIComponent(baseArticle.id)}&x=${requestedCount}&include_llm=${useLlmOutputs}`);
      if (!response.ok) throw new Error('Diverse request failed');
      const data = await response.json();
      const nextResults = data.results || [];
      setDiverseResults(nextResults);
      setChainJobId(data.job_id || null);
      setIsFetchingSummaries(false);
      setIsFetchingDiverse(false);
    } catch (error) {
      console.error('Error fetching diverse options:', error);
      setError('Pri pridobivanju raznolikih clankov je prislo do napake.');
      clearPinTimer();
      setIsPinnedPreview(false);
      setIsClearingResults(false);
    } finally {
      setIsFetchingDiverse(false);
    }
  };

  const handleConfirmDiverse = () => {
    fetchDiverseForArticle(selectedArticle);
  };

  const handleRouteArticleRestart = (article) => {
    fetchDiverseForArticle(article, { pinDelay: 0 });
  };

  const handleTimelessChain = async () => {
    if (!selectedArticle) return;

    setIsFetchingTimelessChain(true);
    setError('');
    setChainJobId(null);
    setIsFetchingSummaries(false);

    try {
      const response = await fetch(`http://127.0.0.1:8000/api/timeless-chain?article_id=${encodeURIComponent(selectedArticle.id)}`);
      if (!response.ok) throw new Error('Timeless chain request failed');
      const data = await response.json();
      const nextResults = data.results || [];

      if (nextResults.length === 0) {
        setError('Ni bilo mogoce ustvariti verige brez casovne omejitve.');
        return;
      }

      setDiverseResults(nextResults);
      setSelectedChainId(nextResults[0].id);
      setChainJobId(data.status === 'complete' ? null : data.job_id || null);
      setIsFetchingSummaries(data.status === 'loading_summaries');
    } catch (error) {
      console.error('Error fetching timeless chain:', error);
      setError('Pri gradnji verige brez casovne omejitve je prislo do napake.');
    } finally {
      setIsFetchingTimelessChain(false);
    }
  };

  const handleChainClick = (chainId) => {
    setSelectedChainId(chainId);
  };

  const handleRouteBack = () => {
    if (selectedChainId) {
      setSelectedChainId(null); // Go back to stage tree
    } else {
      setIsPinnedPreview(false); // Go back to search results
      setIsClearingResults(false);
    }
  };

  const showingRouteStage = isPinnedPreview && selectedArticle;
  const displayedDiverseCount = Math.min(diverseCount, MAX_DIVERSE_COUNT);

  return (
    <div className={`app-container ${!hasSearched ? 'centered-layout' : ''} ${isPinnedPreview ? 'has-pinned-preview' : ''} ${isClearingResults ? 'is-clearing-results' : ''}`}>
      <button
        className="theme-toggle"
        type="button"
        onClick={toggleTheme}
        aria-label={theme === 'dark' ? 'Preklopi na svetel nacin' : 'Preklopi na temen nacin'}
        title={theme === 'dark' ? 'Svetel nacin' : 'Temen nacin'}
      >
        {theme === 'dark' ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="2" />
            <path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        ) : (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M21 14.5A8.5 8.5 0 0 1 9.5 3a7 7 0 1 0 11.5 11.5Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          </svg>
        )}
      </button>

      <div className={`search-section ${hasSearched ? 'fade-out' : 'fade-in'}`}>
        <h1 className="title">Iskalnik Člankov</h1>
        <form onSubmit={handleSearch} className="search-form">
          <div className="search-row">
            <input
              type="text"
              className="query-input"
              placeholder="Vnesite poizvedbo"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button type="submit" className="search-button-small" aria-label="Isci clanke" title="Isci clanke">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
            </button>
          </div>
          
          <div className="slider-container">
            <label htmlFor="k-slider">Število iskanih člankov (k): {k}</label>
            <input
              id="k-slider"
              type="range"
              min="5"
              max="20"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value, 10))}
              className="k-slider"
            />
          </div>
          <button
            type="button"
            className="random-article-button"
            onClick={handleRandomArticle}
            disabled={isRandomLoading || isFetchingDiverse}
          >
            {isRandomLoading ? 'Izbiram...' : 'Naključni članek'}
          </button>
          <div className="random-options">
            <label htmlFor="random-diverse-slider">
              Stevilo verig za nakljucni clanek: {displayedDiverseCount}
            </label>
            <input
              id="random-diverse-slider"
              type="range"
              min="3"
              max={MAX_DIVERSE_COUNT}
              value={displayedDiverseCount}
              onChange={(e) => setDiverseCount(parseInt(e.target.value, 10))}
              className="k-slider"
            />
            <label className="llm-toggle">
              <input
                type="checkbox"
                checked={useLlmOutputs}
                onChange={(e) => setUseLlmOutputs(e.target.checked)}
              />
              <span>Generiraj LLM povezave za verige</span>
            </label>
          </div>
        </form>
      </div>

      {showingRouteStage && (
        <RouteTree
          rootArticle={selectedArticle}
          childArticles={diverseResults}
          isLoading={isFetchingDiverse}
          selectedChainId={selectedChainId}
          onChainClick={handleChainClick}
          isFetchingSummaries={isFetchingSummaries}
          llmEnabled={useLlmOutputs}
          onArticleRestart={handleRouteArticleRestart}
          onTimelessChain={handleTimelessChain}
          isTimelessLoading={isFetchingTimelessChain}
          onBack={handleRouteBack}
        />
      )}

      <div className={`results-section ${hasSearched && !showingRouteStage ? 'fade-in' : 'hidden'}`}>
        <div className="results-header">
          <h2>Rezultati iskanja</h2>
          <button onClick={resetSearch} className="reset-button">Nova poizvedba</button>
        </div>
        {error && <div className="error-message">{error}</div>}
        
        {isSearching ? (
          <div className="loading">Iskanje člankov...</div>
        ) : (
          <div className={`articles-list ${selectedArticle ? 'has-selection' : ''}`}>
            {orderedResults.map((article) => {
              const isSelected = selectedArticle?.id === article.id;

              return (
                <React.Fragment key={article.id}>
                  <ArticleCard
                    ref={isSelected ? selectedAnchorRef : null}
                    article={article}
                    isSelected={isSelected}
                    isDimmed={Boolean(selectedArticle && !isSelected)}
                    onClick={() => handleArticleClick(article)}
                  />
                  {isSelected && !isClearingResults && !isPinnedPreview && (
                    <div className="diverse-controls fade-in">
                      <label htmlFor="diverse-slider">Stevilo raznolikih clankov: {displayedDiverseCount}</label>
                      <input
                        id="diverse-slider"
                        type="range"
                        min="3"
                        max={MAX_DIVERSE_COUNT}
                        value={displayedDiverseCount}
                        onChange={(e) => setDiverseCount(parseInt(e.target.value, 10))}
                        className="k-slider"
                      />
                      <label className="llm-toggle">
                        <input
                          type="checkbox"
                          checked={useLlmOutputs}
                          onChange={(e) => setUseLlmOutputs(e.target.checked)}
                        />
                        <span>Generiraj LLM povezave ob izbiri poti</span>
                      </label>
                      <button
                        onClick={handleConfirmDiverse}
                        className="confirm-diverse-button"
                        disabled={isFetchingDiverse}
                      >
                        {isFetchingDiverse ? 'Iskanje...' : 'Potrdi'}
                      </button>
                    </div>
                  )}
                </React.Fragment>
              );
            })}

            {isFetchingDiverse && (
              <div className="loading">Pridobivanje raznolikih clankov...</div>
            )}

            {!isFetchingDiverse && diverseResults.length > 0 && (
              <div className="diverse-results fade-in">
                <h2>Raznoliki clanki</h2>
                <div className="articles-list">
                  {diverseResults.map((article) => (
                    <ArticleCard
                      key={article.id}
                      article={article}
                      onClick={() => handleArticleClick(article)}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
